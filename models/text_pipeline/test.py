import torch

from preprocess import tokenize_text
from dataset import TextEmotionDataset
from model import BertEmotionClassifier


dataset = TextEmotionDataset()

classes = dataset.label_encoder.classes_


model = BertEmotionClassifier()

model.load_state_dict(
    torch.load(
        "models/text_pipeline/text_model.pth"
    )
)

model.eval()


sample_text = "back"


input_ids, attention_mask = tokenize_text(
    sample_text
)

input_ids = input_ids.unsqueeze(0)

attention_mask = attention_mask.unsqueeze(0)


with torch.no_grad():

    outputs = model(
        input_ids,
        attention_mask
    )

    predicted = torch.argmax(
        outputs,
        dim=1
    )


print("\nInput Text:\n")

print(sample_text)

print("\nPredicted Emotion:\n")

print(classes[predicted.item()])