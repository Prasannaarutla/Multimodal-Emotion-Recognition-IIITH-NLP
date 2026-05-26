import os
import torch
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from preprocess import extract_mfcc


dataset_path = "data/raw"

data = []

max_length = 100


for folder in os.listdir(dataset_path):

    folder_path = os.path.join(
        dataset_path,
        folder
    )

    for file in os.listdir(folder_path):

        if file.endswith(".wav"):

            emotion = folder.replace(
                "OAF_",
                ""
            ).replace(
                "YAF_",
                ""
            ).lower()

            file_path = os.path.join(
                folder_path,
                file
            )

            mfcc = extract_mfcc(file_path)

            mfcc = torch.tensor(mfcc)

            # Padding

            if mfcc.shape[0] < max_length:

                pad_width = max_length - mfcc.shape[0]

                mfcc = torch.nn.functional.pad(
                    mfcc,
                    (0, 0, 0, pad_width)
                )

            else:

                mfcc = mfcc[:max_length]

            data.append({

                "mfcc": mfcc,

                "emotion": emotion
            })


# LABEL ENCODING

df = pd.DataFrame(data)

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(
    df["emotion"]
)


# SAVE FEATURES

torch.save(
    {
        "mfcc": list(df["mfcc"]),
        "labels": list(df["label"]),
        "classes": list(label_encoder.classes_)
    },

    "models/speech_pipeline/processed_data.pt"
)

print("\nProcessed Data Saved Successfully!")