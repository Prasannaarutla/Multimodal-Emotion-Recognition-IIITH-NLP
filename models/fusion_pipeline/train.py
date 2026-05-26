import torch
import torch.nn as nn

from model import MultimodalFusionModel


model = MultimodalFusionModel()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 5


for epoch in range(epochs):

    speech_input = torch.randn(
        32,
        7
    )

    text_input = torch.randn(
        32,
        7
    )

    labels = torch.randint(
        0,
        7,
        (32,)
    )


    optimizer.zero_grad()

    outputs = model(
        speech_input,
        text_input
    )

    loss = criterion(
        outputs,
        labels
    )

    loss.backward()

    optimizer.step()


    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {loss.item():.4f}"
    )


print("\nFusion Training Complete!")


torch.save(
    model.state_dict(),
    "models/fusion_pipeline/fusion_model.pth"
)

print("Fusion Model Saved!")