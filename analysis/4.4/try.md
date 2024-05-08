## What we tried so far

1. Larger Dataset Size
2. Pretrained Weights
3. Reduce Embedding Dimensions (!)
    1. Help GPT model performs better
4. Initialization Matters
    1. Help Transformer performs better
    2. But, the acc is still below 30%


## Analysis

### Model: convnext_custom + gpt + 64

```
model, tokenizer = get_model("convnext_custom", "gpt", {"out_channel": 64}, {"dropout": 0.0, "n_layer": 12, "n_head": 8}, {})
model.load_state_dict(torch.load("./models/latexify-convnext_custom-gpt.pth"))
```

- Good Accuracy + Small Model Size
- Not suitable for auto-regressive task

Problem: When using the model auto-regressively, the model always output the same formula.

Two possible reasons:
1. The CNN is not powerful enough to extract useful features from the images.
2. The attention layer of the decoder fails to attend to the important vectors (the things in the center of the image). As a result, when used auto-regressivly, the model always output the same formula.
    1. I output the results and it seems that the first vector of each batch is always about the same. So, it's more likely that the CNN does not work well.

TODO
1. Investigate the CNN output to check reason 1.