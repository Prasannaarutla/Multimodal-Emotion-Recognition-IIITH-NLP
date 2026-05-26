import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, BertModel


dataset_path = "data/raw"


texts = []

labels = []

speech_features = []


for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(
                root,
                file
            )


            parts = file.replace(
                ".wav",
                ""
            ).split("_")


            if len(parts) >= 3:

                text = parts[1]

                emotion = parts[2]


                if emotion == "pleasant":

                    emotion = "pleasant_surprise"


                texts.append(text)

                labels.append(emotion)


                audio, sr = librosa.load(

                    file_path,

                    sr=22050
                )


                mfcc = librosa.feature.mfcc(

                    y=audio,

                    sr=sr,

                    n_mfcc=40
                )


                mfcc_mean = np.mean(
                    mfcc,
                    axis=1
                )


                speech_features.append(
                    mfcc_mean
                )


tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

bert_model = BertModel.from_pretrained(
    "bert-base-uncased"
)

bert_model.eval()


text_features = []


with torch.no_grad():

    for text in texts:

        encoding = tokenizer(

            text,

            return_tensors="pt",

            padding=True,

            truncation=True,

            max_length=32
        )


        outputs = bert_model(
            **encoding
        )


        cls_embedding = outputs.last_hidden_state[
            :, 0, :
        ].squeeze().numpy()


        text_features.append(
            cls_embedding
        )


speech_features = np.array(
    speech_features
)

text_features = np.array(
    text_features
)


fusion_features = np.concatenate(

    (
        speech_features,

        text_features
    ),

    axis=1
)


label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(
    labels
)


tsne = TSNE(

    n_components=2,

    perplexity=30,

    random_state=42
)


reduced_features = tsne.fit_transform(
    fusion_features
)


plt.figure(figsize=(10, 7))


scatter = plt.scatter(

    reduced_features[:, 0],

    reduced_features[:, 1],

    c=encoded_labels,

    cmap="tab10",

    alpha=0.7
)


plt.xlabel("t-SNE Component 1")

plt.ylabel("t-SNE Component 2")

plt.title("Fusion Representation Visualization")


legend_labels = list(
    label_encoder.classes_
)


plt.legend(

    handles=scatter.legend_elements()[0][:len(legend_labels)],

    labels=legend_labels
)


plt.savefig(
    "Results/plots/fusion_tsne_clusters.png"
)

plt.close()


print(
    "\nFusion t-SNE Visualization Generated Successfully!"
)