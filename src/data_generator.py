import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class VideoDataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=8, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.video_paths = []
        self.labels = []

        self._load_data()
        self.on_epoch_end()

    def _load_data(self):
        categories = ["NonViolence", "Violence"]

        for label, category in enumerate(categories):
            category_path = os.path.join(self.data_dir, category)

            for file in os.listdir(category_path):
                if file.endswith(".npy"):
                    self.video_paths.append(os.path.join(category_path, file))
                    self.labels.append(label)

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = []
        y = []

        for path, label in zip(batch_paths, batch_labels):
            video = np.load(path)   # shape: (20, 96, 96, 3)
            X.append(video)
            y.append(label)

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.video_paths, self.labels))
            np.random.shuffle(temp)
            self.video_paths, self.labels = zip(*temp)

if __name__ == "__main__":
    print("Testing Data Generator...")

    train_gen = VideoDataGenerator(
        data_dir=r"D:\DataScience\Deep Learning Project\Video_Classifier\data\train\processed\train",
        batch_size=4
    )

    print(f"Total batches: {len(train_gen)}")

    X, y = train_gen[0]

    print(f"Batch X shape: {X.shape}")
    print(f"Batch y shape: {y.shape}")