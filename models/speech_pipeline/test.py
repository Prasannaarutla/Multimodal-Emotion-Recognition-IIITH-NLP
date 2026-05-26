import torch

from dataset import SpeechEmotionDataset
from model import BiLSTMEmotionClassifier


dataset = SpeechEmotionDataset(
    "models/speech_pipeline/processed_data.pt"
)

classes = dataset.classes


model = BiLSTMEmotionClassifier()

model.load_state_dict(
    torch.load(
        "models/speech_pipeline/speech_model.pth"
    )
)

model.eval()


sample_mfcc, sample_label = dataset[25]

sample_mfcc = sample_mfcc.unsqueeze(0)


with torch.no_grad():

    outputs = model(sample_mfcc)

    predicted = torch.argmax(
        outputs,
        dim=1
    )


actual_emotion = classes[
    sample_label.item()
]

predicted_emotion = classes[
    predicted.item()
]


print("\nActual Emotion:\n")

print(actual_emotion)

print("\nPredicted Emotion:\n")

print(predicted_emotion)