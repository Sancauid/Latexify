"""
The following code is adapted from
- annotated-transformer: https://github.com/harvardnlp/annotated-transformer
- minGPT: https://github.com/karpathy/minGPT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mingpt import model


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(
        self,
        vocab_size,
        block_size=512,
        n_embd=768,
        n_layer=6,
        n_head=8,
        d_ff=2048,
        dropout=0.1,
        activation="gelu",
    ) -> None:
        super(EncoderDecoder, self).__init__()

        self.block_size = block_size  # for generation as GPT has this attribute
        self.transformer = nn.Transformer(
            d_model=n_embd,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.embed = nn.Sequential(
            Embeddings(n_embd, vocab_size),
            PositionalEncoding(n_embd, dropout, block_size),
        )
        self.generator = Generator(n_embd, vocab_size)

    def forward(self, idx, hidden, targets=None):
        """Take in and process masked src and target sequences.

        Args:
            idx: sequence to the decoder ([B, Ti])
            hidden: sequence of embed vectors to the encoder ([B, Th, D])
            targets: target output ([B, Ti])

        Note:

        1. The code takes in batches of encoded input vectors ([B, Th, D]
        where B is the number of batches, Th is the number of input vectors,
        D is the embed dim).
        2. There is no need to use `src_mask` as all the src input has the same
        size and should all be encoded.
        3. The `tgt_mask` is specified as the square causal mask.
        """
        return self.generator(
            self.transformer.forward(
                hidden,
                self.embed(idx),
                tgt_mask=self.transformer.generate_square_subsequent_mask(idx.size(1)),
                tgt_is_causal=True,
            ),
            targets,
        )


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, embed_dim, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(embed_dim, vocab)

        self.init_weights()

    def forward(self, x, targets=None):
        logits = torch.softmax(self.proj(x), dim=-1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def init_weights(self):
        initrange = 0.1
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)


class Embeddings(nn.Module):
    def __init__(self, embed_size, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, embed_size)
        self.embed_size = embed_size

        self.init_weights()

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.embed_size)

    def init_weights(self):
        initrange = 0.1
        self.lut.weight.data.uniform_(-initrange, initrange)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, embed_size, dropout, block_size):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(block_size, embed_size)
        position = torch.arange(0, block_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        assert x.size(1) <= self.pe.size(
            1
        ), f"Cannot forward sequence of length {x.size(1)}, block size is only {self.pe.size(1)}"

        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def transformer(
    vocab_size,
    block_size=512,
    n_embd=768,
    n_layer=6,
    n_head=8,
    d_ff=2048,
    dropout=0.1,
    activation="gelu",
):
    "Helper: Construct a model from hyperparameters."
    model = EncoderDecoder(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
    )  # model output: logits (dim: [vocab]), loss

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def gpt(vocab_size, block_size=512, n_embd=768, n_layer=6, n_head=8, dropout=0.1):
    """Helper: Construct a GPT Style Decoder Only Model."""
    config = model.GPT.get_default_config()
    config.vocab_size = vocab_size  # type: ignore
    config.block_size = block_size  # type: ignore
    config.n_embd = n_embd  # type: ignore
    config.n_layer = n_layer  # type: ignore
    config.n_head = n_head  # type: ignore
    config.embd_pdrop = dropout  # type: ignore
    config.resid_pdrop = dropout  # type: ignore
    config.attn_pdrop = dropout  # type: ignore
    return model.GPT(config)
