import os
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from preprocess import tokenize_text


class TextEmotionDataset(Dataset):

    def __init__(self):

        dataset_path = "data/raw"

        self.texts = []

        emotions = []


        for folder in os.listdir(dataset_path):

            folder_path = os.path.join(
                dataset_path,
                folder
            )

            emotion = folder.replace(
                "OAF_",
                ""
            ).replace(
                "YAF_",
                ""
            ).lower()

            if emotion == "pleasant_surprised":

                emotion = "pleasant_surprise"


            for file in os.listdir(folder_path):

                if file.endswith(".wav"):

                    parts = file.replace(
                        ".wav",
                        ""
                    ).split("_")

                    if len(parts) >= 2:

                        word = parts[1].lower()

                        self.texts.append(word)

                        emotions.append(emotion)


        self.label_encoder = LabelEncoder()

        self.labels = self.label_encoder.fit_transform(
            emotions
        )


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, idx):

        text = self.texts[idx]

        input_ids, attention_mask = tokenize_text(
            text
        )

        label = torch.tensor(
            self.labels[idx]
        )

        return (
            input_ids,
            attention_mask,
            label
        )


dataset = TextEmotionDataset()

print("\nDataset Size:\n")

print(len(dataset))

sample = dataset[0]

print("\nInput IDs Shape:\n")

print(sample[0].shape)

print("\nEncoded Label:\n")

print(sample[2])