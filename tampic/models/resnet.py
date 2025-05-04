import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock

from .hs_conv import build_hsconv, init_conv


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
        # x: [batch_size, num_channels, height, width]
        # wavelengths: [num_channels], share for all samples in the batch
        batch_size, in_channels, height, width = x.size()
        num_channels = wavelengths.size(0)

        # Map the scalar wavelengths using MLP
        # Let's assume the wavelengths are the same for all samples in the batch, so
        # that we can simply use the first sample
        mapped_wavelengths = (
            self.mlp(wavelengths.view(-1, 1))
            .view(num_channels, self.out_channels, *self.kernel_size)
            .transpose(0, 1)
        )

        # Perform pairwise multiplication with the kernel
        combined_kernels = (
            self.conv.weight.view(self.out_channels, 1, *self.kernel_size)
            * mapped_wavelengths
        )

        # Perform the convolution with the combined kernels
        output = F.conv2d(
            x,
            weight=combined_kernels,
            bias=self.conv.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return output


class _TAMPICConv2d(nn.Conv2d):
    """A wrapper around nn.Conv2d for interface consistency with AdaptiveConvBlock."""

    def forward(self, image: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self, hsi: torch.Tensor, wavelengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hsi: [batch_size, num_channels, height, width]
        # wavelengths: [num_channels], share for all samples in the batch
        b, c, h, w = hsi.size()
        hsi = hsi.permute(0, 2, 3, 1).view(b, h * w, c)
        hsi = self.hsi_module(hsi)
        hsi = hsi.view(b, h, w, self.output_size).permute(0, 3, 1, 2)
        wavelengths = wavelengths.expand(b, 1, c)
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
        *,
        state_dict=None,
        hsi_conv_type: str,
        hsi_avg_dim: int = 462,
        _reuse_head: bool = False,
        _pretrained_hsi_base: bool = False,
        _norm_and_sum: bool = True,  # batch norm and sum
        # _hsi_avg_dim: int | None = None,
    ):
        super().__init__(block, layers, num_classes)
        self._pretrained_hsi_base = _pretrained_hsi_base
        self._norm_and_sum = _norm_and_sum
        self.new_param_name_prefixs = set()

        # if _hsi_avg_dim is not None:
        #     hsi_avg_dim = _hsi_avg_dim
        # self.conv0_hsi = _HSIAvg(hsi_avg_dim)
        # hsi_avg_dim = _hsi_avg_dim
        # self.conv0_hsi = lambda hsi, wavelengths: (hsi, wavelengths)

        self.conv1_hsi = build_hsconv(
            hsi_conv_type,
            in_channels=hsi_avg_dim,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.conv1_target_mask = nn.Conv2d(
            len(self.image_groups), 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # initialization
        if state_dict is not None:  # load pretrained weights
            self.pretrained = True
            if not _reuse_head:
                state_dict.pop("fc.weight", None)
                state_dict.pop("fc.bias", None)
                self.new_param_name_prefixs.update(["conv1_target_mask", "fc"])
            else:
                if self.fc.weight.shape[0] != num_classes:
                    raise ValueError(
                        f"Number of classes in the pretrained model ({self.fc.weight.shape[0]}) "
                        f"does not match the number of classes in the new model ({num_classes})."
                    )
            self.load_state_dict(state_dict, strict=False)
            self.conv1_rgb_red = copy.deepcopy(self.conv1)
            self.conv1_rgb_white = copy.deepcopy(self.conv1)
            # self.conv1_rgb_red._forward = self.conv1_rgb_red.forward
            # self.conv1_rgb_red.forward = (
            #     lambda data, wavelengths: self.conv1_rgb_red._forward(data)
            # )
            # self.conv1_rgb_white._forward = self.conv1_rgb_white.forward
            # self.conv1_rgb_white.forward = (
            #     lambda data, wavelengths: self.conv1_rgb_white._forward(data)
            # )
            if self._pretrained_hsi_base:
                # self.conv1_hsi = _TAMPICConv2d(
                #     hsi_avg_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
                # )
                # # Copy weights from the pretrained conv1 (the first channel) to conv1_hsi
                # # don't do this for bn1_hsi, it is trained from scratch no matter what
                # self.conv1_hsi.weight.data.copy_(self.conv1.weight.data[:, :1])
                # try:
                #     self.new_param_name_prefixs.remove("conv1_hsi")
                # except KeyError:
                #     pass
                self.conv1_hsi.init_conv(self.conv1.weight.data, None)
                self.new_param_name_prefixs.update(
                    [
                        name
                        for name, _ in self.conv1_hsi.named_parameters()
                        if not name.startswith("conv1.")
                    ]
                )
            else:
                # Initialize weights for conv1_hsi
                self.conv1_hsi.init_conv()
                self.new_param_name_prefixs.update(["conv1_hsi"])
            if self._norm_and_sum:
                self.bn1_rgb_red = copy.deepcopy(self.bn1)
                self.bn1_rgb_white = copy.deepcopy(self.bn1)
                self.bn1_hsi = copy.deepcopy(self.bn1)
                self.bn1_target_mask = nn.BatchNorm2d(64)  # from scratch
            else:
                self.bn1_rgb_red = nn.Identity()
                self.bn1_rgb_white = nn.Identity()
                self.bn1_hsi = nn.Identity()
                self.bn1_target_mask = nn.Identity()
            self.new_param_name_prefixs.update(["bn1_target_mask"])
            del self.conv1
            print(f"\tNew param name prefixes: {sorted(self.new_param_name_prefixs)}")
        else:
            self.pretrained = False
            # self.conv1_rgb_red = _TAMPICConv2d(
            self.conv1_rgb_red = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # self.conv1_rgb_white = _TAMPICConv2d(
            self.conv1_rgb_white = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize weights for conv1_rgb_red and conv1_rgb_white
            # nn.init.kaiming_normal_(
            #     self.conv1_rgb_red.weight, mode="fan_out", nonlinearity="relu"
            # )
            # nn.init.kaiming_normal_(
            #     self.conv1_rgb_white.weight, mode="fan_out", nonlinearity="relu"
            # )
            # nn.init.kaiming_normal_(
            #     self.conv1_target_mask.weight, mode="fan_out", nonlinearity="relu"
            # )
            init_conv(self.conv1_rgb_red)
            init_conv(self.conv1_rgb_white)
            self.conv1_hsi.init_conv()
            init_conv(self.conv1_target_mask)
            if self._norm_and_sum:
                self.bn1_rgb_red = nn.BatchNorm2d(64)
                self.bn1_rgb_white = nn.BatchNorm2d(64)
                self.bn1_hsi = nn.BatchNorm2d(64)
                self.bn1_target_mask = nn.BatchNorm2d(64)
            else:
                self.bn1_rgb_red = nn.Identity()
                self.bn1_rgb_white = nn.Identity()
                self.bn1_hsi = nn.Identity()
                self.bn1_target_mask = nn.Identity()
            print("\tAll parameters are new.")

        if self._norm_and_sum:
            self._forward_base = self._forward_base_norm_and_sum
            del self.bn1
        else:
            self._forward_base = self._forward_base_sum_and_norm

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

    @staticmethod
    def get_nonzero(data: dict, image_group: str) -> torch.Tensor:
        return (data[image_group]["available"] & (~data[image_group]["dropped"])).int()

    def _forward_base_norm_and_sum(
        self, data: dict, wavelengths: torch.Tensor
    ) -> torch.Tensor:
        xs = []
        tms = []
        for ig in self.image_groups:  # loop over image groups (modalities)
            avail = self.get_nonzero(data, ig)

            # add target mask
            target_mask = data[ig]["target_mask"]
            target_mask *= avail.view(-1, 1, 1)
            tms.append(target_mask)

            if (avail == 0).all():
                # Skip the image group if it's not available for the entire batch, but
                # target mask is still added above
                continue

            # forward conv layer
            bn1_layer = getattr(self, f"bn1_{ig}".replace("-", "_"))
            conv1_layer = getattr(self, f"conv1_{ig}".replace("-", "_"))
            if ig == "hsi":
                out = bn1_layer(conv1_layer(data[ig]["image"], wavelengths))
            else:
                out = bn1_layer(conv1_layer(data[ig]["image"]))
            out *= avail.view(-1, 1, 1, 1)
            xs.append(out)

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
        data: dict[str, dict],
        wavelengths: torch.Tensor,
        _return_embedding: bool = False,  # backward compatibility
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # NOTE: In the current dataloader implementation, "hsi" will always be a key in
        # data. So strictly speaking we don't have to check if "hsi" is in data.
        # if "hsi" in data and self.get_nonzero(data, "hsi").any():
        #     hsi, wavelengths = self.conv0_hsi(data["hsi"]["image"], wavelengths)
        #     data["hsi"]["image"] = hsi
        last_activation = self._forward_stem(self._forward_base(data, wavelengths))
        embedding = self.avgpool(last_activation).flatten(1)
        logits = self.fc(embedding)
        if _return_embedding:
            return logits, embedding
        else:
            return logits


def resnet_tampic(
    depth: int,
    num_classes: int = 1000,
    pretrained: bool = False,
    hsi_conv_type: str = "gating",
    hsi_avg_dim: int = 462,
    _pretrained_hsi_base: bool = False,
    _norm_and_sum: bool = True,
    # _hsi_avg_dim: int | None = None,
    _reuse_head: bool = False,
):
    """
    Construct a TAMPICResNet model based on the specified ResNet depth.

    Args:
        depth: Depth of ResNet (18, 34, or 50).
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        hsi_avg_dim: Average dimension for hyperspectral imaging.
        _pretrained_hsi_base: Whether to use pretrained weights for the HSI base.
        _norm_and_sum: Whether to normalize and sum spectral bands.
        _reuse_head: If True (only for ResNet-18), keep pretrained head.

    Returns:
        A TAMPICResNet model instance.
    """
    if depth == 18:
        block = BasicBlock
        layers = [2, 2, 2, 2]
        weights_key = "IMAGENET1K_V1"
    elif depth == 34:
        block = BasicBlock
        layers = [3, 4, 6, 3]
        weights_key = "IMAGENET1K_V1"
    elif depth == 50:
        block = BasicBlock
        layers = [3, 4, 6, 3]
        weights_key = "IMAGENET1K_V2"
    else:
        raise ValueError(f"Unsupported ResNet depth: {depth}")

    arch = f"resnet{depth}"
    if pretrained:
        _model = torch.hub.load("pytorch/vision", arch, weights=weights_key)
        state_dict = _model.state_dict()
    else:
        state_dict = None

    return TAMPICResNet(
        block,
        layers,
        num_classes=num_classes,
        state_dict=state_dict,
        hsi_conv_type=hsi_conv_type,
        hsi_avg_dim=hsi_avg_dim,
        _reuse_head=_reuse_head,
        _pretrained_hsi_base=_pretrained_hsi_base,
        _norm_and_sum=_norm_and_sum,
    )
