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


class TAMPICResNet(ResNet):
    image_groups = ["rgb-red", "rgb-white", "hsi"]
    def __init__(
        self, block, layers, num_classes=1000, state_dict=None,
    ):
        super(TAMPICResNet, self).__init__(block, layers, num_classes)

        self.conv1_rgb_red = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.conv1_rgb_white = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.conv1_hsi = AdaptiveConvBlock(
            1, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False
        )
        self.conv1_target_mask = nn.Conv2d(
            len(self.image_groups), 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize weights for conv1_hsi
        nn.init.kaiming_normal_(
            self.conv1_hsi.conv.weight, mode="fan_out", nonlinearity="relu"
        )

        if state_dict is not None:
            self.pretrained = True

            self.load_state_dict(state_dict, strict=False)

            # Copy weights from the pretrained conv1 to conv1_rgb_red and conv1_rgb_white
            self.conv1_rgb_red.weight.data.copy_(self.conv1.weight.data)
            self.conv1_rgb_white.weight.data.copy_(self.conv1.weight.data)

        else:
            self.pretrained = False

        del self.conv1

    def get_param_groups(self) -> tuple[list, list]:
        params_pretrained = []
        params_new = []
        for name, param in self.named_parameters():
            if not self.pretrained:
                params_new.append(param)
            else:
                # hsi base layer, target mask base layer and head are always trained from scratch
                if "conv1_hsi" in name or "fc" in name or "conv1_target_mask" in name:
                    params_new.append(param)
                else:
                    params_pretrained.append(param)
        return params_pretrained, params_new

    def forward(
        self,
        data: dict,
        wavelengths: torch.Tensor,
    ) -> torch.Tensor:
        xs = []
        tms = []
        for ig in self.image_groups:
            # if data[ig]["available"] and not data[ig]["dropped"]:
            if ig == "hsi":
                xs.append(self.conv1_hsi(data[ig]["image"], wavelengths))
            else:
                xs.append(
                    getattr(self, f"conv1_{ig}".replace("-", "_"))(data[ig]["image"])
                )
            tms.append(data[ig]["target_mask"])
        xs.append(self.conv1_target_mask(torch.stack(tms, dim=1)))
        x = torch.stack(xs).sum(0)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18_tampic(num_classes=1000, pretrained=False, _reuse_head: bool = False):
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
        BasicBlock, [2, 2, 2, 2], num_classes=num_classes, state_dict=state_dict
    )
    return model


def resnet34_tampic(num_classes=1000, pretrained=False):
    if pretrained:
        _model = torch.hub.load("pytorch/vision", "resnet34", weights="IMAGENET1K_V1")
        state_dict = _model.state_dict()
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
    else:
        state_dict = None
    model = TAMPICResNet(
        BasicBlock, [3, 4, 6, 3], num_classes=num_classes, state_dict=state_dict
    )

    return model

def resnet50_tampic(num_classes=1000, pretrained=False):
    if pretrained:
        _model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
        state_dict = _model.state_dict()
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
    else:
        state_dict = None
    model = TAMPICResNet(
        BasicBlock, [3, 4, 6, 3], num_classes=num_classes, state_dict=state_dict
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
                "dropped": torch.randint(0, 2, (batch_size, height, width)).bool(),
                "available": torch.randint(0, 2, (batch_size, height, width)).bool(),
                "time_point": torch.randint(0, 10, (batch_size,)),
                "time_points": ["abcd"] * batch_size,
            },
            "rgb-red": {
                "image": torch.randn(batch_size, 3, height, width),
                "target_mask": torch.rand(batch_size, height, width),
                "dropped": torch.randint(0, 2, (batch_size, height, width)).bool(),
                "available": torch.randint(0, 2, (batch_size, height, width)).bool(),
                "time_point": torch.randint(0, 10, (batch_size,)),
                "time_points": ["abcd"] * batch_size,
            },
            "hsi": {
                "image": torch.randn(batch_size, num_channels, height, width),
                "target_mask": torch.rand(batch_size, height, width),
                "dropped": torch.randint(0, 2, (batch_size, height, width)).bool(),
                "available": torch.randint(0, 2, (batch_size, height, width)).bool(),
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
    data["data"]["rgb-red"]["image"] += batch

    prediction = model(data["data"], wavelengths)[0].softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

    # the following is from torchvision docs. Given our model arch, the score should be 
    # the same when inferencing with pretrained weights (41.3%).
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