import torch

from src.encoders.swin import swin_v2_t, swin_v2_s, swin_v2_b


def test_forward():
    BATCH_SIZE = 10
    W = 256
    H = 192
    IN_CHANELS = 3

    x = torch.rand(BATCH_SIZE, IN_CHANELS, H, W)

    model = swin_v2_t()
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 768, 6, 8)

    model = swin_v2_s()
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 768, 6, 8)

    model = swin_v2_b()
    out = model(x)
    assert tuple(out.size()) == (BATCH_SIZE, 1024, 6, 8)
