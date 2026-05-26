import sys

sys.path.append(".")


import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from models.speech_pipeline.dataset import SpeechEmotionDataset


dataset = SpeechEmotionDataset(
    "models/speech_pipeline/processed_data.pt"
)


features = []

labels = []


for mfcc, label in dataset:

    flattened = mfcc.flatten().numpy()

    features.append(flattened)

    labels.append(label.item())


pca = PCA(
    n_components=2
)


reduced_features = pca.fit_transform(
    features
)


label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(
    labels
)


emotion_names = [

    "angry",

    "disgust",

    "fear",

    "happy",

    "neutral",

    "pleasant_surprise",

    "sad"
]


plt.figure(figsize=(10, 7))


scatter = plt.scatter(

    reduced_features[:, 0],

    reduced_features[:, 1],

    c=encoded_labels
)


plt.xlabel("PCA Component 1")

plt.ylabel("PCA Component 2")

plt.title("Speech Emotion Cluster Visualization")


plt.legend(

    handles=scatter.legend_elements()[0],

    labels=emotion_names
)


plt.savefig(
    "Results/plots/pca_emotion_clusters.png"
)

plt.close()


print("\nPCA Visualization Generated Successfully!")