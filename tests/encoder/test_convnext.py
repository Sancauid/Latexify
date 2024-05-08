import torch

from src.encoders.convnext import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
)


def test_forward():
    BATCH_SIZE = 10
    W = 256
    H = 192
    IN_CHANELS = 3

    x = torch.rand(BATCH_SIZE, IN_CHANELS, H, W)

    model = convnext_tiny()
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 768, 6, 8)

    model = convnext_small()
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 768, 6, 8)

    model = convnext_base()
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 1024, 6, 8)

    model = convnext_large()
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 1536, 6, 8)
