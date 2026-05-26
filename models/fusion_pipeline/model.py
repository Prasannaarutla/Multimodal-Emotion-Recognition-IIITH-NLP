import torch
import torch.nn as nn


class MultimodalFusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.speech_fc = nn.Linear(
            7,
            64
        )

        self.text_fc = nn.Linear(
            7,
            64
        )

        self.fusion_fc = nn.Linear(
            128,
            7
        )

        self.relu = nn.ReLU()


    def forward(

        self,

        speech_output,

        text_output
    ):

        speech_features = self.relu(

            self.speech_fc(
                speech_output
            )
        )

        text_features = self.relu(

            self.text_fc(
                text_output
            )
        )

        combined = torch.cat(

            (
                speech_features,
                text_features
            ),

            dim=1
        )

        output = self.fusion_fc(
            combined
        )
