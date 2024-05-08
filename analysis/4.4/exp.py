import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(sys.path[0]), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data_handler import Im2LatexDataHandler
from src.data_loader import get_data_loaders
from src.generate import beam_search
from src.train import get_model


def imshow(image, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


model, tokenizer = get_model(
    "convnext_custom",
    "gpt",
    {"out_channel": 64},
    {"dropout": 0.0, "n_layer": 12, "n_head": 8},
    {"file_path": "../../data/dataset5/step2/dict_id2word.pkl"},
)
model.load_state_dict(
    torch.load("../latexify-convnext_custom-64-gpt.pth", map_location="cpu")
)

handler = Im2LatexDataHandler(data_dir="../../data/dataset5", train_percentage=0.0001)
df_combined, y_combined, tuple_len = handler.load_data_and_images()
train_loader = get_data_loaders(
    df_combined,
    y_combined,
    "train",
    tokenizer,
    tuple_len,
    batch_size=1,
)
image, x, y = next(iter(train_loader))

imshow(image[0].cpu().numpy())
plt.show()

print(f"target length: {len(y[0])}")
print(f"target: {tokenizer.decode(y[0])}")

idx = beam_search(model, image, tokenizer, beam_width=2)
predicted = tokenizer.decode(idx)
print(f"predict: {predicted}")
