import torch

from model import MultimodalFusionModel


model = MultimodalFusionModel()

model.load_state_dict(
    torch.load(
        "models/fusion_pipeline/fusion_model.pth"
    )
)

model.eval()


speech_prediction = torch.tensor([
    [0.90, 0.02, 0.01, 0.02, 0.01, 0.02, 0.02]
])

text_prediction = torch.tensor([
    [0.85, 0.03, 0.02, 0.03, 0.02, 0.03, 0.02]
])


with torch.no_grad():

    output = model(
        speech_prediction,
        text_prediction
    )

    predicted = torch.argmax(
        output,
        dim=1
    )


classes = [

    "angry",

    "disgust",

    "fear",

    "happy",

    "neutral",

    "pleasant_surprise",

    "sad"
]


print("\nFusion Output:\n")

print(output)

print("\nFinal Predicted Emotion:\n")

print(classes[predicted.item()])