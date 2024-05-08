import torch

from src.encoders.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def test_forward():
    BATCH_SIZE = 10
    W = 256
    H = 192
    IN_CHANELS = 3

    x = torch.rand(BATCH_SIZE, IN_CHANELS, H, W)

    model = resnet18(in_channels=IN_CHANELS)
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 512, 6, 8)

    model = resnet34(in_channels=IN_CHANELS)
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 512, 6, 8)

    model = resnet50(in_channels=IN_CHANELS)
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 2048, 6, 8)

    model = resnet101(in_channels=IN_CHANELS)
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 2048, 6, 8)

    model = resnet152(in_channels=IN_CHANELS)
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 2048, 6, 8)
