import torch

from src.encoders.convnext import convnext_base
from src.decoders.transformer import gpt, transformer
from src.models import LatexifyModel


def test_forward_with_encdec():
    # encoder
    BATCH_SIZE = 10
    W = 256
    H = 192
    IN_CHANELS = 3

    # decoder
    VOCAB_SIZE = 64
    BLOCK_SIZE = 256  # for pos encoding (should > SEQ_SIZE)
    EMBED_DIM = 1024
    N_LAYER = 6
    N_HEAD = 8
    D_FF = 2048
    DROPOUT = 0.1
    SEQ_LEN = 1

    images = torch.rand(BATCH_SIZE, IN_CHANELS, H, W)
    idx = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))

    model = LatexifyModel(
        convnext_base(),
        transformer(
            VOCAB_SIZE,
            block_size=BLOCK_SIZE,
            n_embd=EMBED_DIM,
            n_layer=N_LAYER,
            n_head=N_HEAD,
            d_ff=D_FF,
            dropout=DROPOUT,
        ),
        "convnext_base-transformer",
    )
    out, loss = model(idx, images)
    assert loss is None
    assert tuple(out.size()) == (
        BATCH_SIZE,
        SEQ_LEN,
        VOCAB_SIZE,
    )  # 1 is the length of the seq


def test_forward_with_dec():
    # encoder
    BATCH_SIZE = 10
    W = 256
    H = 192
    IN_CHANELS = 3

    # decoder
    VOCAB_SIZE = 64
    BLOCK_SIZE = 256  # for pos encoding (should > SEQ_SIZE)
    EMBED_DIM = 1024
    N_LAYER = 6
    N_HEAD = 8

    images = torch.rand(BATCH_SIZE, IN_CHANELS, H, W)
    idx = torch.tensor([[]])

    model = LatexifyModel(
        convnext_base(),
        gpt(
            VOCAB_SIZE,
            block_size=BLOCK_SIZE,
            n_embd=EMBED_DIM,
            n_layer=N_LAYER,
            n_head=N_HEAD,
        ),
        "convnext_base-gpt",
    )
    out, loss = model(idx, images)
    assert loss is None
    assert tuple(out.size()) == (
        BATCH_SIZE,
        6 * 8,
        VOCAB_SIZE,
    )  # 6*8 is the number of embed vectors extracted from images
