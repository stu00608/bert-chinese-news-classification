import os
import numpy as np
from datasets import Dataset, load_metric
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer)

path = "./data/toutiao_cat_data.txt"

label2tag = {
    0: "news_story",
    1: "news_culture",
    2: "news_entertainment",
    3: "news_sports",
    4: "news_finance",
    5: "news_house",
    6: "news_car",
    7: "news_edu",
    8: "news_tech",
    9: "news_military",
    10: "news_travel",
    11: "news_world",
    12: "stock",
    13: "news_agriculture",
    14: "news_game",
}
tag2label = {v: k for k, v in label2tag.items()}
n_labels = len(label2tag)
max_length = 30


def tokenize_fn(examples):
    return tokenizer(
        examples["texts"],
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        "voidful/albert_chinese_tiny", model_max_length=max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        "voidful/albert_chinese_tiny",
        num_labels=n_labels,
    )

    # TODO: Adjustable export path
    # TODO: TrainingArguments setting.
    training_args = TrainingArguments(
        output_dir="test_exp", evaluation_strategy="epoch")
    metric = load_metric("accuracy")

    texts = []
    labels = []

    with open(path) as f:
        for line in f.readlines():
            split = line.split("_!_")
            text = split[3]
            label = tag2label[split[2]]
            texts.append(text)
            labels.append(label)

    dataset = Dataset.from_dict({"texts": texts, "labels": labels})

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
