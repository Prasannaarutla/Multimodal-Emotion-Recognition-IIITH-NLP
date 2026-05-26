import torch

from torch.utils.data import Dataset


class SpeechEmotionDataset(Dataset):

    def __init__(self, data_path):

        data = torch.load(data_path)

        self.mfccs = data["mfcc"]

        self.labels = data["labels"]

        self.classes = data["classes"]

    def __len__(self):

        return len(self.mfccs)

    def __getitem__(self, idx):

        mfcc = self.mfccs[idx].float()

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return mfcc, label


# TESTING

dataset = SpeechEmotionDataset(
    "models/speech_pipeline/processed_data.pt"
)

print("\nDataset Size:\n")

print(len(dataset))

sample_mfcc, sample_label = dataset[0]

print("\nMFCC Shape:\n")

print(sample_mfcc.shape)

print("\nEncoded Label:\n")

print(sample_label)

