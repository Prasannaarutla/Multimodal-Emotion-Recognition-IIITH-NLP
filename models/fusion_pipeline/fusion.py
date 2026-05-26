import torch

from model import MultimodalFusionModel


# LOAD TRAINED FUSION MODEL

model = MultimodalFusionModel()

model.load_state_dict(

    torch.load(
        "models/fusion_pipeline/fusion_model.pth"
    )
)

model.eval()


# DUMMY SPEECH OUTPUT
# (simulating speech model prediction)

speech_output = torch.randn(
    1,
    6
)


# DUMMY TEXT OUTPUT
# (simulating text model prediction)

text_output = torch.randn(
    1,
    6
)


# EMOTION LABELS

emotion_labels = [

    "angry",

    "disgust",

    "fear",

    "happy",

    "neutral",

    "sad"
]


# FINAL FUSION PREDICTION

with torch.no_grad():

    output = model(

        speech_output,

        text_output
    )

    predicted_class = torch.argmax(

        output,

        dim=1
    ).item()


# PRINT FINAL RESULT

print("\nFinal Emotion Prediction:\n")

print(

    emotion_labels[
        predicted_class
    ]
)