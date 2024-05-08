import torch

from src.decoders.transformer import gpt


def test_forward():
    # input
    BATCH_SIZE = 10
    ENCODED_VECTORS = 100
    SEQ_SIZE = 128

    # model
    VOCAB_SIZE = 512
    BLOCK_SIZE = 256  # for pos encoding (should > SEQ_SIZE)
    EMBED_DIM = 64
    N_LAYER = 6
    N_HEAD = 8

    hidden = torch.rand(BATCH_SIZE, ENCODED_VECTORS, EMBED_DIM)
    idx = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_SIZE))

    model = gpt(
        VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_embd=EMBED_DIM,
        n_layer=N_LAYER,
        n_head=N_HEAD,
    )

    _, loss = model(idx, hidden)
    assert loss is None

    targets = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_SIZE))
    _, loss = model(idx, hidden, targets=targets)
