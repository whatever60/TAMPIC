import math

import torch
from torch import nn
from torch.nn import functional as F


class HSConvBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias

    def aap(
        self, x: torch.Tensor, wavelengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive average pooling on x and wavelengths to match in_channels.

        Args:
            x: [batch_size, num_channels, height, width]
            wavelengths: [batch_size, num_channels]

        Returns:
            x_pooled: [batch_size, in_channels, height, width]
            wavelengths_pooled: [batch_size, in_channels]
        """
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        x_pooled = F.adaptive_avg_pool1d(x_flat, self.in_channels)
        x_pooled = x_pooled.view(b, h, w, self.in_channels).permute(0, 3, 1, 2)

        wavelengths_pooled = F.adaptive_avg_pool1d(wavelengths, self.in_channels)
        return x_pooled, wavelengths_pooled

    def init_conv(
        self,
        conv_weight_data: torch.Tensor | None = None,
        conv_bias_data: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize convolution weights.

        Args:
            conv_weight_data: Predefined weights for the convolution.
        """
        conv = getattr(self, "conv", None)
        if conv is None:
            return
        init_conv(conv, conv_weight_data, conv_bias_data)


def init_conv(
    conv: nn.Conv2d,
    conv_weight_data: torch.Tensor | None = None,
    conv_bias_data: torch.Tensor | None = None,
) -> None:
    """
    Initialize convolution weights.

    Args:
        conv: Convolution layer to initialize.
        conv_weight_data: Predefined weights for the convolution.
    """
    if conv_weight_data is None:  # Initialize weights using nn.init
        nn.init.kaiming_uniform_(conv.weight, a=0, mode="fan_in", nonlinearity="relu")
        # nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
    else:
        conv_weight_data = conv_weight_data.mean(dim=1, keepdim=True).expand(
            -1, conv.weight.shape[1], -1, -1
        )
        conv.weight.data.copy_(conv_weight_data)

    if conv.bias is not None:
        if conv_bias_data is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(conv.bias, -bound, bound)
        else:
            conv.bias.data.copy_(conv_bias_data)


class HSConvIdentity(HSConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        x, wavelengths = self.aap(x, wavelengths)
        return x


class HSConvPlain(HSConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        x, wavelengths = self.aap(x, wavelengths)
        return self.conv(x)


class HSConvAdaptive(HSConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv = nn.Conv2d(
            1, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )

        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels * kernel_size[0] * kernel_size[1]),
            nn.LayerNorm(out_channels * kernel_size[0] * kernel_size[1]),
        )

    def forward(self, x: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x, wavelengths = self.aap(x, wavelengths)
        # shape: (B, C_out, C_in, kH, kW)
        mapped_kernel = (
            self.mlp(wavelengths.unsqueeze(-1))
            .view(b, self.in_channels, self.out_channels, *self.kernel_size)
            .transpose(1, 2)
        )
        # Combine learned weights with base conv weights (elementwise modulation)
        base_kernel = self.conv.weight
        # shape: (B, C_out, C_in, kH, kW)
        combined_kernel = base_kernel * mapped_kernel

        x = x.reshape(1, b * self.in_channels, h, w)
        # NOTE: shape of tensors in grouped conv
        # - x: [batch_size, num_groups * in_channels_per_group, height, width]
        # - weight：[num_groups * out_channels_per_group, in_channels_per_group, kH, kW]
        # - bias: [num_groups * out_channels_per_group]
        combined_kernel = combined_kernel.flatten(end_dim=1)
        bias = self.conv.bias.repeat(b)

        output = F.conv2d(
            x,
            combined_kernel,
            bias=bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=b,
        )
        # Reshape back to (B, C_out, H_out, W_out)
        output = output.view(b, self.out_channels, output.shape[-2], output.shape[-1])
        return output

    def forward_slow(self, x: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x, wavelengths = self.aap(x, wavelengths)
        mapped_kernel = self.mlp(wavelengths.view(b * self.in_channels, 1)).view(
            b, self.in_channels, self.out_channels, *self.kernel_size
        )
        base_kernel = self.conv.weight.transpose(0, 1)
        # [batch_size, in_channels, out_channels, kH, kW]
        combined_kernel = base_kernel * mapped_kernel
        # [batch_size, out_channels, in_channels, kH, kW]
        combined_kernel = combined_kernel.transpose(1, 2)
        out = []
        for i in range(b):
            out.append(
                F.conv2d(
                    x[i].unsqueeze(0),
                    combined_kernel[i],
                    bias=self.conv.bias,
                    stride=self.conv.stride,
                    padding=self.conv.padding,
                ).squeeze(0)
            )
        out = torch.stack(out, dim=0)
        return out


class HSConvGating(HSConvBase):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        hidden_dims: list[int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.hidden_dims = hidden_dims
        final_hidden = hidden_dims[-1]

        self.proj_x = self._make_mlp(
            in_channels, hidden_dims, final_hidden, activation_last=False
        )
        self.proj_gate = self._make_mlp(
            in_channels, hidden_dims, final_hidden, activation_last=True
        )

        self.ff = nn.Sequential(
            nn.Linear(final_hidden, final_hidden * 4),
            nn.ReLU(),
            nn.Linear(final_hidden * 4, final_hidden),
        )
        self.post_norm = nn.LayerNorm(final_hidden)

        self.conv = nn.Conv2d(
            final_hidden,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def _make_mlp(
        self, in_dim: int, hidden_dims: list[int], out_dim: int, activation_last: bool
    ) -> nn.Sequential:
        layers = []
        dims = [in_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], out_dim))
        if activation_last:
            layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x, wavelengths = self.aap(x, wavelengths)
        x = self.proj_x(x.permute(2, 3, 0, 1))  # [h, w, b, hidden_dims[-1]]
        gate = self.proj_gate(wavelengths)  # [b, hidden_dims[-1]]
        # [h, w, b, hidden_dims[-1]]
        x = x * gate
        x = self.post_norm(self.ff(x) + x)
        return self.conv(x)  # [b, out_channels, h0, w0]


def build_hsconv(
    conv_type: str,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
    hidden_dims: list[int] | None = [128, 32],
) -> nn.Module:
    """
    Factory for hyperspectral convolution modules with consistent interface.

    Args:
        conv_type: One of {"identity", "plain", "adaptive", "gating"}.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Tuple of (height, width) for convolution kernel.
        stride: Stride for convolution.
        padding: Padding for convolution.
        bias: Whether to use a bias in conv layers.
        hidden_dims: Only for "gating"; MLP hidden dimensions.

    Returns:
        An nn.Module implementing the chosen conv variant.
    """
    conv_type = conv_type.lower()
    if conv_type == "identity":
        return HSConvIdentity(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif conv_type == "plain":
        return HSConvPlain(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif conv_type == "adaptive":
        return HSConvAdaptive(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_dims=hidden_dims,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif conv_type == "gating":
        if hidden_dims is None:
            raise ValueError("hidden_dims must be provided for 'gating'")
        return HSConvGating(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_dims=hidden_dims,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    else:
        raise ValueError(
            f"Unknown conv_type '{conv_type}'. Valid options: identity, plain, adaptive, gating."
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, in_channels, height, width = 5, 17, 224, 224

    x = torch.randn(batch_size, in_channels, height, width)
    wavelengths = torch.randn(batch_size, in_channels)

    print("✅ HSConvPlain")
    plain = HSConvPlain(
        in_channels=7, out_channels=11, kernel_size=(7, 7), padding=3, stride=2
    )
    out = plain(x, wavelengths)
    print("Output shape:", out.shape)

    print("\n✅ HSConvAdaptive")
    adaptive = HSConvAdaptive(
        in_channels=7, out_channels=11, kernel_size=(7, 7), padding=3, stride=2
    )
    out = adaptive(x, wavelengths)
    out_slow = adaptive.forward_slow(x, wavelengths)
    print("Output shape:", out.shape)
    assert torch.allclose(out, out_slow, atol=1e-5), "Outputs do not match!"

    print("\n✅ HSConvGating")
    gating = HSConvGating(
        in_channels=7,
        hidden_dims=[128, 64],
        out_channels=11,
        kernel_size=(7, 7),
        padding=3,
        stride=2,
    )
    out = gating(x, wavelengths)
    print("Output shape:", out.shape)
