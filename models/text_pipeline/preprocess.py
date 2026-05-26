from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)


def tokenize_text(text):

    encoding = tokenizer(

        text,

        padding="max_length",

        truncation=True,

        max_length=32,

        return_tensors="pt"
    )

    return (
        encoding["input_ids"].squeeze(0),
        encoding["attention_mask"].squeeze(0)
    )