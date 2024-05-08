# Implement functions to generate tokens and convert them back to the latex
#
# Reference
# - annotated-transformer: https://nlp.seas.harvard.edu/annotated-transformer/#inference
# - minGPT: https://github.com/karpathy/minGPT
# - Simple implementation of beam search in Python:
#   https://hussainwali.medium.com/simple-implementation-of-beam-search-in-python-64b2d3e2fd7e

import torch
from torch.nn import functional as F


@torch.inference_mode()
def generate(
    model,
    idx,
    tokenizer,
    max_new_tokens=200,
    temperature=1.0,
    do_sample=False,
    top_k=None,
):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.

    Note:

    1. For GPT decoder, when idx is empty, it should be initialized as `torch.tensor([[]])`.
    """
    model.eval()

    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = (
            idx if idx.size(1) <= model.block_size else idx[:, -model.block_size :]
        )
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

        if idx_next is tokenizer.end_token_id:
            return idx[:-1]

    return idx


@torch.inference_mode()
def beam_search(model, image, tokenizer, beam_width=2, max_new_tokens=200):
    """Beam Search.

    Args:
        images: [B, C, H, W] where B = 1.
    """
    model.eval()

    if tokenizer.start_token_id is None:
        idx = torch.tensor([[]], device=image.device)
    else:
        idx = torch.tensor([[tokenizer.start_token_id]], device=image.device)

    beam = [(idx, 1)]

    for _ in range(max_new_tokens):
        candidates = []
        for idx, prob in beam:
            logits, _ = model(idx, image)
            logits = logits[:, -1, :]
            top_k_prob, top_k_idx = torch.topk(logits, beam_width)
            for k_val, k_idx in zip(top_k_prob[0], top_k_idx[0]):
                if idx.size(1) == 0:
                    candidates.append((torch.tensor([[k_idx]]), prob * k_val))
                else:
                    candidates.append(
                        (
                            torch.cat([idx, torch.tensor([[k_idx]])], dim=-1),
                            prob * k_val,
                        )
                    )

        # Select the top `beam_width` candidates based on their scores
        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # if the top one has already reached the end, then we should return it directly
        if beam[0][0][0, -1].item() == tokenizer.end_token_id:
            return beam[0][0][0, :-1]

    # Return the sequence with the highest score
    return max(beam, key=lambda x: x[1])[0][0][0]
