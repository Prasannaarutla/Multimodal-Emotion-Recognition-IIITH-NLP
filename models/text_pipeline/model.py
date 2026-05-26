import torch
import torch.nn as nn

from transformers import BertModel


class BertEmotionClassifier(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            768,
            num_classes
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(
            pooled_output
        )

        logits = self.fc(
            pooled_output
        )

        return logits