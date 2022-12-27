import os
import wandb
import numpy as np
from datasets import Dataset, load_metric
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          TrainingArguments,
                          Trainer,
                          set_seed)

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
seed = 1520
runs_path = "./runs"
model_name = "voidful/albert_chinese_tiny"


def tokenize_fn(examples):
    global tokenizer
    return tokenizer(
        examples["texts"],
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True)


def compute_metrics(eval_pred):
    global metric
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main(dataset, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(
        kwargs["model_name"],
        num_labels=n_labels,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(runs_path, kwargs["name"]),
        evaluation_strategy="epoch",
        num_train_epochs=kwargs["epoch"],
        per_device_train_batch_size=kwargs["batch_size"],
        run_name=kwargs["name"]
    )

    all_dataset = dataset
    all_dataset = all_dataset.map(tokenize_fn, batched=True)

    if kwargs["data_size"] != 0:
        dataset = Dataset.from_dict(dataset[:kwargs["data_size"]])

    dataset = dataset.map(tokenize_fn, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=all_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=max_length)

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
    dataset = dataset.shuffle(seed=seed)

    metric = load_metric("accuracy")

    data_size = 200000
    batch_size = 8
    epoch = 20

    main(dataset, name=f"exp_val_all_e_{epoch}_b_{batch_size}_d_{data_size}",
         data_size=data_size, epoch=epoch, batch_size=batch_size, model_name=model_name)
    wandb.finish()