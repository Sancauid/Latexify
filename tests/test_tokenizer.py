from src.tokenizer import Tokenizer


def test_tokenizer_encode():
    tokens = [0, 1, 2, 3]

    tokenizer = Tokenizer(use_gpt=False)
    assert list(tokenizer.encode(tokens)) == tokens + [tokenizer.pad_token_id] * (
        tokenizer.get_max_label_length() - len(tokens)
    )

    # vocab_size - 2 is used as the start token
    tokenizer = Tokenizer(use_gpt=True)
    assert list(tokenizer.encode(tokens)) == [
        tokenizer.get_vocab_size() - 1
    ] + tokens + [tokenizer.pad_token_id] * (
        tokenizer.get_max_label_length() - len(tokens)
    )


def test_tokenizer_decode():
    formulas = [
        32,
        305,
        334,
        315,
        304,
        334,
        15,
        336,
        327,
        309,
        336,
        28,
        150,
        334,
        16,
        336,
        334,
        260,
        334,
        17,
        336,
        336,
        179,
        298,
        305,
        334,
        327,
        309,
        336,
        45,
        305,
        334,
        47,
        336,
        9,
        298,
        305,
        334,
        309,
        327,
        336,
        304,
        334,
        131,
        336,
        45,
        305,
        334,
        41,
        336,
        244,
        212,
        334,
        309,
        322,
        326,
        336,
        96,
        146,
        150,
        334,
        314,
        260,
        334,
        320,
        305,
        334,
        327,
        336,
        320,
        305,
        334,
        309,
        336,
        336,
        336,
        334,
        17,
        42,
        305,
        334,
        52,
        336,
        336,
        6,
        125,
        305,
        334,
        47,
        336,
        45,
        305,
        334,
        47,
        336,
        9,
        125,
        305,
        334,
        41,
        336,
        45,
        305,
        334,
        41,
        336,
        7,
        60,
        60,
        60,
        60,
        13,
    ]
    tokenizer = Tokenizer(use_gpt=False)

    assert (
        tokenizer.decode(formulas)
        == r"C_{h^{0}tc}=\frac{1}{\sqrt{2}}\left[\xi_{tc}P_{R}+\xi_{ct}^{\dagger}P_{L}\right]\operatorname{cos}\alpha\equiv\frac{g\sqrt{m_{t}m_{c}}}{2M_{W}}(\chi_{R}P_{R}+\chi_{L}P_{L})\,\,\,\,."
    )
