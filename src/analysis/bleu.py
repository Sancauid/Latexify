# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from torcheval BLEU implementation.
#
# Modify several parts to match the wikipedia description.
# 1. Brevity Penalty: https://en.wikipedia.org/wiki/BLEU#Brevity_penalty

from collections import Counter as counter
from typing import Counter, Optional, Sequence, Tuple

import torch


def _bleu_score_update(
    input: Sequence[Sequence[str]],
    target: Sequence[Sequence[Sequence[str]]],
    n_gram: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ = input
    target_ = target

    if len(input_) != len(target_):
        raise ValueError(
            f"Input and target corpus should have same sizes, but input corpus size = {len(input_)}, target corpus size = {len(target_)} "
        )

    input_len = torch.tensor(0, device=device)
    target_len = torch.tensor(0, device=device)
    matches_by_order = torch.zeros(n_gram, device=device)
    possible_matches_by_order = torch.zeros(n_gram, device=device)

    for candidate, references in zip(input_, target_):
        candidate_tokenized = candidate
        references_tokenized = references

        len_candidate = len(candidate_tokenized)
        # See `closest_ref_length` in https://www.nltk.org/_modules/nltk/translate/bleu_score.html
        len_reference = min(
            (len(ref) for ref in references_tokenized),
            key=lambda ref_len: (abs(ref_len - len_candidate), ref_len),
        )  # effective reference corpus length
        input_len += len_candidate
        target_len += len_reference

        # this calculates the modified n-gram precision
        # https://en.wikipedia.org/wiki/BLEU#Modified_n-gram_precision
        candidate_ngram_counter = _get_ngrams(candidate_tokenized, n_gram)
        reference_ngram_counter = counter()
        for ref in references_tokenized:
            reference_ngram_counter |= _get_ngrams(ref, n_gram)
        overlap = candidate_ngram_counter & reference_ngram_counter

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]

        # this is equal to the denominator
        for i in range(n_gram):
            if len_candidate - i > 0:
                possible_matches_by_order[i] += len_candidate - i

    return input_len, target_len, matches_by_order, possible_matches_by_order


def _bleu_score_compute(
    input_len: torch.Tensor,
    target_len: torch.Tensor,
    matches_by_order: torch.Tensor,
    possible_matches_by_order: torch.Tensor,
    n_gram: int,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weights is not None and n_gram != weights.size(dim=0):
        raise ValueError(
            f"the length of weights should equal n_gram, got len(weights)={weights.size(dim=0)}, n_gram={n_gram}"
        )

    if weights is None:
        weights = torch.tensor([1 / n_gram] * n_gram, device=matches_by_order.device)

    precisions = matches_by_order / possible_matches_by_order
    geometric_mean = torch.exp(torch.sum(weights * torch.log(precisions)))

    brevity_penalty = _calc_brevity_penalty(input_len, target_len)

    return brevity_penalty * geometric_mean


def _calc_brevity_penalty(
    input_len: torch.Tensor, target_len: torch.Tensor
) -> torch.Tensor:
    if input_len > target_len:
        return torch.tensor(1.0, device=input_len.device)
    else:
        return torch.exp(1 - target_len / input_len)


def _get_ngrams(sentence: Sequence[str], n_gram: int) -> Counter[str]:
    """
    Args:
        sentence: text from which we get n-grams
        n_gram: length of n-gram
    """
    if n_gram not in [1, 2, 3, 4]:
        raise ValueError(f"n_gram should be 1, 2, 3, or 4, got {n_gram}.")
    ngram_counts = counter()
    for n_val in range(1, n_gram + 1):
        for i in range(0, len(sentence) - n_val + 1):
            ngram = tuple(sentence[i : i + n_val])
            ngram_counts[ngram] += 1
    return ngram_counts


class BLEUScore:
    """
    Compute BLEU score (https://en.wikipedia.org/wiki/BLEU) given translations and references.
    Its functional version is ``torcheval.metrics.functional.text.bleu``.

    Args:
        n_gram: Maximum n-gram to use when computing BLEU score. Can be 1, 2, 3, or 4.
        weights: Optional weight distribution of n-grams. Requires len(weights) = n_gram. If unspecified, will use uniform weights.

    Examples:
        >>> import torch
        >>> from torcheval.metrics import BLEUScore
        >>> metric = BLEUScore(n_gram=4)
        >>> candidates = ["the squirrel is eating the nut", "the cat is on the mat"]
        >>> references = [["a squirrel is eating a nut", "the squirrel is eating a tasty nut"], ["there is a cat on the mat", "a cat is on the mat"]]
        >>> metric.update(candidates, references)
        >>> metric.compute()
        tensor(0.65341892)
        >>> candidates = ["i like ice cream and apple pie"]
        >>> references = [["i like apple pie with ice cream on top", "i like ice cream with my apple pie", "i enjoy my apple pie with ice cream"]]
        >>> metric.update(candidates, references)
        >>> metric.compute()
        tensor(0.56377503)
    """

    def __init__(
        self,
        *,
        n_gram: int,
        weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if n_gram not in [1, 2, 3, 4]:
            raise ValueError(f"n_gram should be 1, 2, 3, or 4, got {n_gram}.")
        if weights is not None and n_gram != len(weights):
            raise ValueError(
                f"the length of weights should equal n_gram, got len(weights)={len(weights)}, n_gram={n_gram}"
            )

        self.weights = weights
        self.n_gram = n_gram
        self.device = device

        self.input_len = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.target_len = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.matches_by_order = torch.zeros(n_gram, dtype=torch.float64, device=device)
        self.possible_matches_by_order = torch.zeros(
            n_gram, dtype=torch.float64, device=device
        )

    @torch.inference_mode()
    def update(
        self,
        input: Sequence[Sequence[str]],
        target: Sequence[Sequence[Sequence[str]]],
    ):
        """
        Update the metric state with new inputs.

        Args:
            input: Translations to score.
            target: List of references for each translation.
        """
        (
            input_len,
            target_len,
            matches_by_order,
            possible_matches_by_order,
        ) = _bleu_score_update(input, target, self.n_gram, self.device)
        self.input_len += input_len
        self.target_len += target_len
        self.matches_by_order += matches_by_order
        self.possible_matches_by_order += possible_matches_by_order
        return self

    @torch.inference_mode()
    def compute(self) -> torch.Tensor:
        """
        Returns the running BLEUScore. If no ``update()`` calls are made before
        ``compute()`` is called, return tensor(0.0).
        """
        if torch.sum(self.matches_by_order) == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=self.device)
        return _bleu_score_compute(
            self.input_len,
            self.target_len,
            self.matches_by_order,
            self.possible_matches_by_order,
            self.n_gram,
            self.weights,
        )
