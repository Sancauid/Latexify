import os
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor, Pad
from src.tokenizer import Tokenizer
import gc  # Garbage Collector interface
import numpy as np


class PreProcessing:
    def __init__(self, data_dir="./data/dataset5"):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "formula_images")
        self.training_dir = os.path.join(data_dir, "training_56")
        self.input_data_dir = os.path.join(data_dir, "training_56")
        self.batch_size = 14000
        self.tokenizer_transformer = Tokenizer(True)
        self.tokenizer_gpt = Tokenizer(False)

    def load_data(self, filename):
        """Load DataFrame from a pickle file."""
        file_path = os.path.join(self.input_data_dir, filename)
        return pd.read_pickle(file_path)

    def load_image(self, image_filename):
        """Load an image file."""
        image_path = os.path.join(self.image_dir, image_filename)
        return Image.open(image_path)

    def load_data_batches(self, dataframe, batch_size):
        """Generator to load images in batches."""

        for start in range(0, len(dataframe), batch_size):
            new_rows = []
            end = start + batch_size
            batch = dataframe[start:end]

            for _, row in batch.iterrows():

                name = row["image"]
                label = row["word2id"]
                # PAD IMAGE
                image = self.load_image(name)
                image = image.convert("RGB")
                to_tensor = ToTensor()
                image = to_tensor(image)
                pad_height = 128 - image.size()[1]
                pad_width = 1088 - image.size()[2]
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                padding = (pad_left, pad_top, pad_right, pad_bottom)
                pad_transform = Pad(padding, fill=1, padding_mode="constant")
                image = pad_transform(image)
                label_gpt = self.tokenizer_gpt.encode(label)
                label_transformer = self.tokenizer_transformer.encode(label)
                new_row = {
                    "name": name,
                    "image": image,
                    "label_gpt": label_gpt,
                    "label_transformer": label_transformer,
                }
                new_rows.append(new_row)

            new_batch_df = pd.DataFrame(new_rows)
            pickle_filename = f"batch_{start}_{batch_size}.pkl"
            new_batch_df.to_pickle(pickle_filename)
            print(pickle_filename)
            del new_rows, new_batch_df  # Delete large variables
            gc.collect()

    def preprocessing_data(self):
        """Load all Data and Images into a single DataFrame and list."""
        df_train = self.load_data("df_train.pkl")
        # df_test = self.load_data("df_test.pkl")
        # df_valid = self.load_data("df_valid.pkl")

        df_train = df_train.drop_duplicates(subset="image", keep="first")
        # df_test = df_test.drop_duplicates(subset="image", keep="first")
        # df_valid = df_valid.drop_duplicates(subset="image", keep="first")

        df_train_processes = pd.DataFrame(
            {"name": [], "image": [], "label_gpt": [], "label_transformer": []}
        )

        self.load_data_batches(df_train, self.batch_size)

        df_train_processes.reset_index(drop=True, inplace=True)

        pickle_filename = "df_train_processes.pkl"
        df_train_processes.to_pickle(pickle_filename)

        """
        df_test_processes = pd.DataFrame({
            "name":[],
            "image": [],
            "label_gpt": [],
            "label_transformer":[]
        })

        for new_batch_df in self.load_data_batches(df_test,self.batch_size):
            df_test_processes = pd.concat([df_test_processes,new_batch_df])

        df_test_processes.reset_index(drop=True, inplace=True)

        pickle_filename = 'df_test_processes.pkl'
        df_test_processes.to_pickle(pickle_filename)



        df_valid_processes = pd.DataFrame({
            "name":[],
            "image": [],
            "label_gpt": [],
            "label_transformer":[]
        })

        for new_batch_df in self.load_data_batches(df_valid,self.batch_size):
            df_valid_processes = pd.concat([df_valid_processes,new_batch_df])

        df_valid_processes.reset_index(drop=True, inplace=True)

        pickle_filename = 'df_valid_processes.pkl'
        df_valid_processes.to_pickle(pickle_filename)


        return
        """
