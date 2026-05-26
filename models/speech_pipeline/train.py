import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import SpeechEmotionDataset
from model import BiLSTMEmotionClassifier


dataset = SpeechEmotionDataset(
    "models/speech_pipeline/processed_data.pt"
)

indices = list(range(len(dataset)))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_subset = torch.utils.data.Subset(
    dataset,
    train_indices
)

test_subset = torch.utils.data.Subset(
    dataset,
    test_indices
)

train_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_subset,
    batch_size=64,
    shuffle=False
)

model = BiLSTMEmotionClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

for epoch in range(epochs):

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    for mfcc, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(mfcc)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )

print("\nTraining Complete!")

torch.save(
    model.state_dict(),
    "models/speech_pipeline/speech_model.pth"
)

print("Model Saved!")