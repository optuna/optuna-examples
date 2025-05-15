"""
Optuna example for fine-tuning a BERT-based text classification model on the IMDb dataset
with hyperparameter optimization using Optuna. In this example, we fine-tune a lightweight
pre-trained BERT model on a small subset of the IMDb dataset to classify movie reviews as
positive or negative. We optimize the validation accuracy by tuning the learning rate, batch size
and number of training epochs.
"""

from datasets import load_dataset
import evaluate
import torch

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import set_seed
from transformers import Trainer
from transformers import TrainingArguments


set_seed(42)


device = torch.device("cpu")


raw_dataset = load_dataset("imdb")

raw_train = raw_dataset["train"].shuffle(seed=42).select(range(1000))
raw_test = raw_dataset["test"].shuffle(seed=42).select(range(500))

model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)


tokenized_train = raw_train.map(tokenize, batched=True).select_columns(["input_ids", "label"])
tokenized_test = raw_test.map(tokenize, batched=True).select_columns(["input_ids", "label"])


metric = evaluate.load("accuracy")


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=-1)
    labels = eval_pred.label_ids
    return metric.compute(predictions=predictions, references=labels)


def compute_objective(metrics):
    return metrics["eval_accuracy"]


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
    report_to="none",
    fp16=False,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64, 128]
        ),
    }


# Use Trainer's built-in hyperparameter tuning with Optuna
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=10,
    compute_objective=compute_objective,
)

# Print best result
print("Best trial:")
print(best_run)
