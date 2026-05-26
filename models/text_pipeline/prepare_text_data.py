import os
import pandas as pd


dataset_path = "data/raw"

texts = []

labels = []


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

                texts.append(word)

                labels.append(emotion)


df = pd.DataFrame({

    "text": texts,

    "emotion": labels
})


print(df.head())

print("\nTotal Samples:\n")

print(len(df))

print("\nEmotion Counts:\n")

print(df["emotion"].value_counts())