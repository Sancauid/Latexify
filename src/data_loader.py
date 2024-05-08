from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class CustomLatexDataset(Dataset):
    def __init__(self, df, y, split, tokenizer, tuple_len):
        assert split in {"train", "test", "validation"}

        if split == "train":
            self.start_split = 0
            self.end_split = tuple_len[0]
        elif split == "test":
            self.start_split = tuple_len[0]
            self.end_split = tuple_len[0] + tuple_len[1]
        else:  # validation
            self.start_split = tuple_len[0] + tuple_len[1]
            self.end_split = tuple_len[0] + tuple_len[1] + tuple_len[2]

        self.split = split
        self.image_df = df.reset_index(drop=True)
        self.imgs_y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return self.end_split - self.start_split

    def __getitem__(self, idx):
        # print(idx)
        img_label = self.image_df.iloc[self.start_split + idx]["word2id"]
        name = self.image_df.iloc[self.start_split + idx]["image"]

        tok_label = self.tokenizer.encode(img_label)

        if not self.tokenizer.use_gpt:
            y = tok_label[1:].clone().detach()
            x = tok_label[:-1]
        else:
            y = tok_label.clone().detach()
            x = tok_label
        y[y == self.tokenizer.pad_token_id] = -1  # ignore_index

        image = self.imgs_y[name]
        to_tensor = ToTensor()
        image = to_tensor(image)

        return image, x, y


def get_data_loaders(
    df_combined, y_combined, split, tokenizer, tuple_len, batch_size=56
):
    current_dataset = CustomLatexDataset(
        df_combined, y_combined, split, tokenizer, tuple_len
    )

    current_dataset_loader = DataLoader(
        current_dataset, batch_size=batch_size, shuffle=True
    )

    return current_dataset_loader
