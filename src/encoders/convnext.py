from torch import nn
from torchvision import models


def convnext_custom(out_channel=256, **kwargs):
    """Adapted from convnext_tiny."""
    assert (
        out_channel % 8 == 0
    ), f"Cannot construct model of out_channel {out_channel}, out_channel must be divisible by 8"
    out1 = out_channel // 8
    out2 = out_channel // 4
    out3 = out_channel // 2
    block_setting = [
        models.convnext.CNBlockConfig(out1, out2, 3),
        models.convnext.CNBlockConfig(out2, out3, 3),
        models.convnext.CNBlockConfig(out3, out_channel, 9),
        models.convnext.CNBlockConfig(out_channel, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    model = models.convnext._convnext(
        block_setting,
        stochastic_depth_prob,
        **{"weights": None, "progress": True, **kwargs},
    )
    model.avgpool = nn.Identity()  # type: ignore
    model.classifier = nn.Identity()  # type: ignore
    return model


def convnext_tiny(**kwargs):
    model = models.convnext_tiny(**kwargs)
    model.avgpool = nn.Identity()  # type: ignore
    model.classifier = nn.Identity()  # type: ignore
    return model


def convnext_small(**kwargs):
    model = models.convnext_small(**kwargs)
    model.avgpool = nn.Identity()  # type: ignore
    model.classifier = nn.Identity()  # type: ignore
    return model


def convnext_base(**kwargs):
    model = models.convnext_base(**kwargs)
    model.avgpool = nn.Identity()  # type: ignore
    model.classifier = nn.Identity()  # type: ignore
    return model


def convnext_large(**kwargs):
    model = models.convnext_large(**kwargs)
    model.avgpool = nn.Identity()  # type: ignore
    model.classifier = nn.Identity()  # type: ignore
    return model
