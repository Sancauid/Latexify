from src.analysis import BLEUScore


def test_corpus_bleu():
    candidate_corpus = [["My", "full", "pytorch", "test"], ["Another", "Sentence"]]
    references_corpus = [
        [["My", "full", "pytorch", "test"], ["Completely", "Different"]],
        [["No", "Match"]],
    ]
    bleu = BLEUScore(n_gram=4)
    bleu.update(candidate_corpus, references_corpus)
    assert bleu.compute().item() == 0.8408964152537145


def test_corpus_bleu_nltk():
    # Examples from nltk
    hyp1 = [
        "It",
        "is",
        "a",
        "guide",
        "to",
        "action",
        "which",
        "ensures",
        "that",
        "the",
        "military",
        "always",
        "obeys",
        "the",
        "commands",
        "of",
        "the",
        "party",
    ]
    ref1a = [
        "It",
        "is",
        "a",
        "guide",
        "to",
        "action",
        "that",
        "ensures",
        "that",
        "the",
        "military",
        "will",
        "forever",
        "heed",
        "Party",
        "commands",
    ]
    ref1b = [
        "It",
        "is",
        "the",
        "guiding",
        "principle",
        "which",
        "guarantees",
        "the",
        "military",
        "forces",
        "always",
        "being",
        "under",
        "the",
        "command",
        "of",
        "the",
        "Party",
    ]
    ref1c = [
        "It",
        "is",
        "the",
        "practical",
        "guide",
        "for",
        "the",
        "army",
        "always",
        "to",
        "heed",
        "the",
        "directions",
        "of",
        "the",
        "party",
    ]
    hyp2 = [
        "he",
        "read",
        "the",
        "book",
        "because",
        "he",
        "was",
        "interested",
        "in",
        "world",
        "history",
    ]
    ref2a = [
        "he",
        "was",
        "interested",
        "in",
        "world",
        "history",
        "because",
        "he",
        "read",
        "the",
        "book",
    ]

    references_corpus = [[ref1a, ref1b, ref1c], [ref2a]]
    candidate_corpus = [hyp1, hyp2]
    bleu = BLEUScore(n_gram=4)
    bleu.update(candidate_corpus, references_corpus)

    assert bleu.compute().item() == 0.5920778868801042
