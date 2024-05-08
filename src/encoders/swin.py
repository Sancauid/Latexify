from torch import nn
from torchvision import models


def swin_v2_t(**kwargs):
    model = models.swin_v2_t(**kwargs)
    model.avgpool = nn.Identity()  # type: ignore
    model.flatten = nn.Identity()  # type: ignore
    model.head = nn.Identity()  # type: ignore
    return model


def swin_v2_s(**kwargs):
    model = models.swin_v2_s(**kwargs)
    model.avgpool = nn.Identity()  # type: ignore
    model.flatten = nn.Identity()  # type: ignore
    model.head = nn.Identity()  # type: ignore
    return model


def swin_v2_b(**kwargs):
    model = models.swin_v2_b(**kwargs)
    model.avgpool = nn.Identity()  # type: ignore
    model.flatten = nn.Identity()  # type: ignore
    model.head = nn.Identity()  # type: ignore
    return model
