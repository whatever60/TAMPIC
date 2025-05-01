import argparse
import time

import torch
from torchvision.io import read_image
from torchvision.models import ResNet18_Weights
import lightning as L

from resnet import AdaptiveConvBlock, resnet18_tampic  # adjust import path


def run_test() -> None:
    """Run unit tests and example inference for resnet18_tampic.

    Includes basic architecture checks, dummy data tests, and
    an example classification on an image.
    """
    L.seed_everything(42, workers=True)
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

    model = resnet18_tampic(num_classes=30, pretrained=False)
    print(model(data["data"], wavelengths).size())

    model = resnet18_tampic(num_classes=30, pretrained=True)
    print(model(data["data"], wavelengths).size())

    model = resnet18_tampic(num_classes=30, pretrained=True, _hsi_avg_dim=8)
    print(model(data["data"], wavelengths).size())

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

    model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
    model.eval()
    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")


def run_benchmark(device: str = "cuda", drop_hsi: bool = True) -> None:
    """Run one training step (forward + backward + optimizer) and time it.

    Args:
        device: Device to run the benchmark on. 'cuda' or 'cpu'.
    """
    L.seed_everything(42, workers=True)

    batch_size, num_channels, height, width = 64, 462, 96, 96
    wavelengths = torch.linspace(-1, 1, num_channels).to(device)

    data = {
        "rgb-white": {
            "image": torch.randn(batch_size, 3, height, width, device=device),
            "target_mask": torch.rand(batch_size, height, width, device=device),
            "dropped": torch.zeros(
                batch_size, device=device
            ).bool(),  # always not dropped
            "available": torch.ones(
                batch_size, device=device
            ).bool(),  # always available
            "time_point": torch.randint(0, 10, (batch_size,), device=device),
            "time_points": ["abcd"] * batch_size,
        },
        "rgb-red": {
            "image": torch.randn(batch_size, 3, height, width, device=device),
            "target_mask": torch.rand(batch_size, height, width, device=device),
            "dropped": torch.zeros(
                batch_size, device=device
            ).bool(),  # always not dropped
            "available": torch.ones(
                batch_size, device=device
            ).bool(),  # always available
            "time_point": torch.randint(0, 10, (batch_size,), device=device),
            "time_points": ["abcd"] * batch_size,
        },
        "hsi": {
            "image": torch.randn(
                batch_size, num_channels, height, width, device=device
            ),
            "target_mask": torch.rand(batch_size, height, width, device=device),
            "dropped": torch.full((batch_size,), drop_hsi, device=device).bool(),
            "available": torch.ones(batch_size, device=device).bool(),
            "time_point": torch.randint(0, 10, (batch_size,), device=device),
            "time_points": ["abcd"] * batch_size,
            "hsi_channel_dropout": torch.randint(
                0, 2, (batch_size, num_channels), device=device
            ).bool(),
        },
    }

    model = resnet18_tampic(num_classes=30, pretrained=True, _hsi_avg_dim=None).to(
        device
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    labels = torch.randint(0, 30, (batch_size,), device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = model(data, wavelengths)
    loss = criterion(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(f"Training step time: {time.perf_counter() - t0:.3f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TAMPIC test or benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("test", help="Run test inference and architecture checks")

    parser_benchmark = subparsers.add_parser(
        "benchmark", help="Run speed benchmark of training step"
    )
    parser_benchmark.add_argument(
        "--device", type=str, default="cpu", help="Device to run on (cuda or cpu)"
    )
    parser_benchmark.add_argument(
        "--drop-hsi",
        action="store_true",
        help="Whether to drop HSI data during benchmark",
    )

    args = parser.parse_args()

    if args.command == "test":
        run_test()
    elif args.command == "benchmark":
        run_benchmark(device=args.device, drop_hsi=args.drop_hsi)
