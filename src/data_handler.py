import os
import pandas as pd
from PIL import Image


class Im2LatexDataHandler:
    def __init__(self, data_dir="./data/dataset5", train_percentage=0.5):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images-post")
        self.training_dir = os.path.join(data_dir, "training_56")
        self.input_data_dir = os.path.join(data_dir, "training_56")
        self.batch_size = 56
        self.train_percentage = train_percentage

    def load_data(self, filename):
        """Load DataFrame from a pickle file."""
        file_path = os.path.join(self.input_data_dir, filename)
        return pd.read_pickle(file_path)

    def load_image(self, image_filename):
        """Load an image file."""
        image_path = os.path.join(self.image_dir, image_filename)
        return Image.open(image_path)

    def load_images_in_batches(self, dataframe):
        """Generator to load images in batches."""
        batch = {}
        for image_filename in dataframe["image"]:
            img = self.load_image(image_filename)
            batch[image_filename] = img
            if len(batch) == self.batch_size:
                yield batch
                batch = {}
        if batch:
            yield batch

    def load_data_and_images(self):
        df_train = self.load_data("df_train.pkl")
        df_valid = self.load_data("df_valid.pkl")
        df_test = self.load_data("df_test.pkl")

        df_train = df_train.iloc[0 : int(len(df_train.index) * self.train_percentage)]

        df_train = df_train.drop_duplicates(subset="image", keep="first")
        df_test = df_test.drop_duplicates(subset="image", keep="first")
        df_valid = df_valid.drop_duplicates(subset="image", keep="first")

        df_combined = pd.concat([df_train, df_valid, df_test]).reset_index(drop=True)

        df_combined = df_combined.drop_duplicates(subset="image", keep="first")
        y_combined = {}
        for batch in self.load_images_in_batches(df_combined):
            y_combined.update(batch)

        if not (df_combined.index == range(df_combined.shape[0])).all():
            print("Index is not continuous or does not start from 0.")
            df_combined = df_combined.reset_index(drop=True)
            print("Index after reset:", df_combined.index)

        print(len(df_combined), len(y_combined))

        return df_combined, y_combined, (len(df_train), len(df_test), len(df_valid))
