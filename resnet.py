import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models.resnet import ResNet, BasicBlock


class AdaptiveConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super(AdaptiveConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Convolutional layer with one input channel
        self.conv = nn.Conv2d(
            1, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )

        # MLP for scalar mapping with layer normalization
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels * kernel_size[0] * kernel_size[1]),
            nn.LayerNorm(out_channels * kernel_size[0] * kernel_size[1]),
        )

    def forward(self, x, wavelengths):
        batch_size, in_channels, height, width = x.size()

        # Map the scalar wavelengths using MLP
        # Let's assume the wavelengths are the same for all samples in the batch, so
        # that we can simply use the first sample
        mapped_wavelengths = self.mlp(wavelengths.view(-1, 1)).view(
            self.out_channels, -1, *self.kernel_size
        )

        # Perform pairwise multiplication with the kernel
        combined_kernels = (
            self.conv.weight.view(self.out_channels, 1, *self.kernel_size)
            * mapped_wavelengths
        )

        # Perform the convolution with the combined kernels
        output = F.conv2d(
            x,
            combined_kernels,
            bias=self.conv.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return output


class _TAMPICConv2d(nn.Conv2d):
    def forward(self, image: torch.tensor, wavelengths: torch.Tensor) -> torch.tensor:
        return super().forward(image)


class _HSIAvg(nn.Module):
    """This is a module without learnable parameters that averages the HSI data across 
    channels given output size utilizing adaptive 1d pooling. It also takes 
    corresponding average for the wavelengths.
    """
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.hsi_module = nn.AdaptiveAvgPool1d(output_size)
        self.wavelength_module = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, hsi: torch.Tensor, wavelengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hsi: [batch_size, num_channels, height, width]
        # wavelengths: [batch_size, num_channels]
        b, c, h, w = hsi.size()
        hsi = hsi.permute(0, 2, 3, 1).view(b, h * w, c)
        hsi = self.hsi_module(hsi)
        hsi = hsi.view(b, h, w, self.output_size).permute(0, 3, 1, 2)
        wavelengths = wavelengths.view(b, 1, c)
        wavelengths = self.wavelength_module(wavelengths)
        wavelengths = wavelengths.view(b, self.output_size)
        return hsi, wavelengths


class TAMPICResNet(ResNet):
    image_groups = ["rgb-red", "rgb-white", "hsi"]

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        state_dict=None,
        num_hsi_channels: int = 462,
        _pretrained_hsi_base: bool = None,
        _norm_and_sum: bool = True,  # batch norm and sum
        _hsi_avg_dim: int | None = None,
    ):
        super(TAMPICResNet, self).__init__(block, layers, num_classes)
        self._pretrained_hsi_base = _pretrained_hsi_base
        self._norm_and_sum = _norm_and_sum
        self.new_param_name_prefixs = set()

        if _hsi_avg_dim is not None:
            self.conv0_hsi = _HSIAvg(_hsi_avg_dim)
            num_hsi_channels = _hsi_avg_dim
        else:
            self.conv0_hsi = lambda hsi, wavelengths: (hsi, wavelengths)

        self.conv1_hsi = AdaptiveConvBlock(
            1, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False
        )
        self.conv1_target_mask = nn.Conv2d(
            len(self.image_groups), 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if self._norm_and_sum:
            self.bn1_hsi = nn.BatchNorm2d(64)
            self.bn1_target_mask = nn.BatchNorm2d(64)
            self.new_param_name_prefixs.update(["bn1_hsi", "bn1_target_mask"])
        else:
            self.bn1_hsi = (None,)
            self.bn1_target_mask = None
        nn.init.kaiming_normal_(
            self.conv1_hsi.conv.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.conv1_target_mask.weight, mode="fan_out", nonlinearity="relu"
        )
        self.new_param_name_prefixs.update(["conv1_hsi", "conv1_target_mask"])

        if state_dict is not None:
            self.pretrained = True
            self.load_state_dict(state_dict, strict=False)
            self.conv1_rgb_red = copy.deepcopy(self.conv1)
            self.conv1_rgb_white = copy.deepcopy(self.conv1)
            self.conv1_rgb_red._forward = self.conv1_rgb_red.forward
            self.conv1_rgb_red.forward = (
                lambda data, wavelengths: self.conv1_rgb_red._forward(data)
            )
            self.conv1_rgb_white._forward = self.conv1_rgb_white.forward
            self.conv1_rgb_white.forward = (
                lambda data, wavelengths: self.conv1_rgb_white._forward(data)
            )
            if self._norm_and_sum:
                self.bn1_rgb_red = copy.deepcopy(self.bn1)
                self.bn1_rgb_white = copy.deepcopy(self.bn1)
            else:
                self.bn1_rgb_red = None
                self.bn1_rgb_white = None
            if self._pretrained_hsi_base:
                self.conv1_hsi = _TAMPICConv2d(
                    num_hsi_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                # Copy weights from the pretrained conv1 (the first channel) to conv1_hsi
                # don't do thi for bn1_hsi, it is trained from scratch no matter what
                self.conv1_hsi.weight.data.copy_(self.conv1.weight.data[:, :1])
                try:
                    self.new_param_name_prefixs.remove("conv1_hsi")
                except KeyError:
                    pass
            else:
                # Initialize weights for conv1_hsi
                nn.init.kaiming_normal_(
                    self.conv1_hsi.conv.weight, mode="fan_out", nonlinearity="relu"
                )
        else:
            self.pretrained = False
            self.conv1_rgb_red = _TAMPICConv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.conv1_rgb_white = _TAMPICConv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if self._norm_and_sum:
                self.bn1_rgb_red = nn.BatchNorm2d(64)
                self.bn1_rgb_white = nn.BatchNorm2d(64)
                self.new_param_name_prefixs.update(["bn1_rgb_red", "bn1_rgb_white"])
            else:
                self.bn1_rgb_red = None
                self.bn1_rgb_white = None
            # Initialize weights for conv1_rgb_red and conv1_rgb_white
            nn.init.kaiming_normal_(
                self.conv1_rgb_red.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.kaiming_normal_(
                self.conv1_rgb_white.weight, mode="fan_out", nonlinearity="relu"
            )
            self.new_param_name_prefixs.update(["conv1_rgb_red", "conv1_rgb_white"])

        del self.conv1
        if self._norm_and_sum:
            self._forward_base = self._forward_base_norm_and_sum
            del self.bn1
        else:
            self._forward_base = self._forward_base_sum_and_norm

        print(f"\tNew param name prefixes: {sorted(self.new_param_name_prefixs)}")

    def get_param_groups(self) -> tuple[list, list]:
        params_pretrained = []
        params_new = []
        for name, param in self.named_parameters():
            if (
                any(
                    name.startswith(f"{prefix}.")
                    for prefix in self.new_param_name_prefixs
                )
                or not self.pretrained
            ):
                params_new.append(param)
            else:
                params_pretrained.append(param)
        return params_pretrained, params_new

    def _forward_base_norm_and_sum(
        self, data: dict, wavelengths: torch.Tensor
    ) -> torch.Tensor:
        xs = []
        tms = []
        for ig in self.image_groups:
            dont_zero = (data[ig]["available"] & (~data[ig]["dropped"])).int()
            # forward conv layer
            bn1_layer = getattr(self, f"bn1_{ig}".replace("-", "_"))
            # if ig == "hsi":
            #     out = bn1_layer(self.conv1_hsi(data[ig]["image"], wavelengths))
            # else:
            conv1_layer = getattr(self, f"conv1_{ig}".replace("-", "_"))
            out = bn1_layer(conv1_layer(data[ig]["image"], wavelengths))
            out *= dont_zero.view(-1, 1, 1, 1)
            xs.append(out)
            # add target mask
            target_mask = data[ig]["target_mask"]
            target_mask *= dont_zero.view(-1, 1, 1)
            tms.append(target_mask)
        xs.append(self.bn1_target_mask(self.conv1_target_mask(torch.stack(tms, dim=1))))
        x = torch.stack(xs).sum(0)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _forward_base_sum_and_norm(
        self, data: dict, wavelengths: torch.Tensor
    ) -> torch.Tensor:
        xs = []
        tms = []
        for ig in self.image_groups:
            # if data[ig]["available"] and not data[ig]["dropped"]:
            # if ig == "hsi":
            #     xs.append(self.conv1_hsi(data[ig]["image"], wavelengths))
            # else:
            xs.append(
                getattr(self, f"conv1_{ig}".replace("-", "_"))(
                    data[ig]["image"], wavelengths
                )
            )
            tms.append(data[ig]["target_mask"])
        xs.append(self.conv1_target_mask(torch.stack(tms, dim=1)))
        x = torch.stack(xs).sum(0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(
        self,
        data: dict,
        wavelengths: torch.Tensor,
    ) -> torch.Tensor:
        if "hsi" in data:
            hsi, wavelengths = self.conv0_hsi(data["hsi"]["image"], wavelengths)
            data["hsi"]["image"] = hsi
        return self.fc(
            self.avgpool(
                self._forward_stem(self._forward_base(data, wavelengths))
            ).flatten(1)
        )


def resnet18_tampic(
    num_classes=1000,
    pretrained=False,
    num_wavelengths: int = 462,
    _pretrained_hsi_base=False,
    _norm_and_sum=True,
    _hsi_avg_dim: int | None = None,
    _reuse_head: bool = False,
):
    # if pretrained:
    #     state_dict = torch.hub.load_state_dict_from_url(
    #         "https://download.pytorch.org/models/resnet18-5c106cde.pth", progress=True
    #     )
    # else:
    #     state_dict = None
    # model = TAMPICResNet(
    #     BasicBlock, [2, 2, 2, 2], num_classes=num_classes, weights=ResNet18_Weights.IMAGENET1K_V1
    # )
    if pretrained:
        _model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
        state_dict = _model.state_dict()
        if not _reuse_head:
            del state_dict["fc.weight"]
            del state_dict["fc.bias"]
    else:
        state_dict = None
    model = TAMPICResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        state_dict=state_dict,
        num_hsi_channels=num_wavelengths,
        _pretrained_hsi_base=_pretrained_hsi_base,
        _norm_and_sum=_norm_and_sum,
        _hsi_avg_dim=_hsi_avg_dim,
    )
    return model


def resnet34_tampic(
    num_classes=1000,
    pretrained=False,
    num_wavelengths: int = 462,
    _pretrained_hsi_base=False,
    _norm_and_sum=True,
    _hsi_avg_dim: int | None = None,
):
    if pretrained:
        _model = torch.hub.load("pytorch/vision", "resnet34", weights="IMAGENET1K_V1")
        state_dict = _model.state_dict()
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
    else:
        state_dict = None
    model = TAMPICResNet(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        state_dict=state_dict,
        num_hsi_channels=num_wavelengths,
        _pretrained_hsi_base=_pretrained_hsi_base,
        _norm_and_sum=_norm_and_sum,
        _hsi_avg_dim=_hsi_avg_dim,
    )

    return model


def resnet50_tampic(
    num_classes=1000,
    pretrained=False,
    num_wavelengths: int = 462,
    _pretrained_hsi_base=False,
    _norm_and_sum=True,
    _hsi_avg_dim: int | None = None,
):
    if pretrained:
        _model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
        state_dict = _model.state_dict()
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
    else:
        state_dict = None
    model = TAMPICResNet(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        state_dict=state_dict,
        num_hsi_channels=num_wavelengths,
        _pretrained_hsi_base=_pretrained_hsi_base,
        _norm_and_sum=_norm_and_sum,
        _hsi_avg_dim=_hsi_avg_dim,
    )

    return model


if __name__ == "__main__":
    from torchvision.io import read_image
    from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights

    # test adaptive conv block
    batch_size, num_channels, height, width = 2, 6, 224, 224
    out_channels = 64
    kernel_size = (7, 7)
    stride = 2
    padding = 3
    x = torch.randn(batch_size, num_channels, height, width)
    wavelengths = torch.rand(num_channels)
    block = AdaptiveConvBlock(num_channels, out_channels, kernel_size, stride, padding)
    output = block(x, wavelengths)
    assert output.size() == (batch_size, out_channels, height // 2, width // 2)
    print(output.size())

    # Dummy input
    # ideally dropped can only be True where available is True, but for testing purposes
    # we will ignore this constraint
    data = {
        "data": {
            "rgb-white": {
                "image": torch.randn(batch_size, 3, height, width),
                "target_mask": torch.rand(batch_size, height, width),
                "dropped": torch.randint(0, 2, (batch_size,)).bool(),
                "available": torch.randint(0, 2, (batch_size,)).bool(),
                "time_point": torch.randint(0, 10, (batch_size,)),
                "time_points": ["abcd"] * batch_size,
            },
            "rgb-red": {
                "image": torch.randn(batch_size, 3, height, width),
                "target_mask": torch.rand(batch_size, height, width),
                "dropped": torch.randint(0, 2, (batch_size,)).bool(),
                "available": torch.randint(0, 2, (batch_size,)).bool(),
                "time_point": torch.randint(0, 10, (batch_size,)),
                "time_points": ["abcd"] * batch_size,
            },
            "hsi": {
                "image": torch.randn(batch_size, num_channels, height, width),
                "target_mask": torch.rand(batch_size, height, width),
                "dropped": torch.randint(0, 2, (batch_size,)).bool(),
                "available": torch.randint(0, 2, (batch_size,)).bool(),
                "time_point": torch.randint(0, 10, (batch_size,)),
                "time_points": ["abcd"] * batch_size,
                "hsi_channel_dropout": torch.randint(
                    0, 2, (batch_size, num_channels)
                ).bool(),
            },
        },
        "label": 2,
    }
    wavelengths = torch.linspace(-1, 1, num_channels)

    # from scratch
    model = resnet18_tampic(num_classes=30, pretrained=False)
    output = model(data["data"], wavelengths)
    _ = model.get_param_groups()
    print(output.size())

    # pretrained
    model = resnet18_tampic(num_classes=30, pretrained=True)
    output = model(data["data"], wavelengths)
    _ = model.get_param_groups()
    print(output.size())

    # pretrained with hsi avg
    model = resnet18_tampic(num_classes=30, pretrained=True, _hsi_avg_dim=8)
    output = model(data["data"], wavelengths)
    _ = model.get_param_groups()
    print(output.size())

    # test with an image
    model = resnet18_tampic(num_classes=1000, pretrained=True, _reuse_head=True)
    model.eval()
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    img = read_image("test_data/kitten.jpg")
    batch = transform(img).unsqueeze(0)

    data = data.copy()
    for ig in data["data"]:
        data["data"][ig]["image"] *= 0
        data["data"][ig]["target_mask"] *= 0
        data["data"][ig]["available"] = torch.ones_like(data["data"][ig]["available"])
        data["data"][ig]["dropped"] = torch.ones_like(data["data"][ig]["dropped"])
    data["data"]["rgb-red"]["image"] += batch
    data["data"]["rgb-red"]["available"] = torch.ones_like(
        data["data"]["rgb-red"]["available"]
    )
    data["data"]["rgb-red"]["dropped"] = torch.zeros_like(
        data["data"]["rgb-red"]["dropped"]
    )

    prediction = model(data["data"], wavelengths)[0].softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

    # the following is from torchvision docs. Given our model arch, the score should be
    # the same when using pretrained weights (41.3%).
    # Step 1: Initialize model with the best available weights
    weights = ResNet18_Weights.DEFAULT
    model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
    model.eval()
    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)
    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")
