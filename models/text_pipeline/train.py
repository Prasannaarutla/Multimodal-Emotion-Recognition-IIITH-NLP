import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import TextEmotionDataset
from model import BertEmotionClassifier


dataset = TextEmotionDataset()

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
    batch_size=4,
    shuffle=True
)

test_loader = DataLoader(
    test_subset,
    batch_size=4,
    shuffle=False
)

model = BertEmotionClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-5
)

epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    for input_ids, attention_mask, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(
            input_ids,
            attention_mask
        )

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

print("\nText Training Complete!")

torch.save(
    model.state_dict(),
    "models/text_pipeline/text_model.pth"
)

print("Text Model Saved!")