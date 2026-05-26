import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, BertModel


dataset_path = "data/raw"


texts = []

labels = []


for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".wav"):

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


tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

bert_model = BertModel.from_pretrained(
    "bert-base-uncased"
)

bert_model.eval()


features = []


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


        features.append(
            cls_embedding
        )


features = np.array(features)


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
    features
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

plt.title("t-SNE Contextual Representation Visualization")


legend_labels = list(
    label_encoder.classes_
)


plt.legend(

    handles=scatter.legend_elements()[0][:len(legend_labels)],

    labels=legend_labels
)


plt.savefig(
    "Results/plots/text_tsne_clusters.png"
)

plt.close()


print(
    "\nt-SNE Text Visualization Generated Successfully!"
)