"""
Optuna example for fine-tuning a BERT-based text classification model on the IMDb dataset
with hyperparameter optimization using Optuna. In this example, we fine-tune a lightweight
pre-trained BERT model on a small subset of the IMDb dataset to classify movie reviews as
positive or negative.We optimize the validation accuracy by tuning the learning rate, batch size
and number of training epochs.
"""

from datasets import load_dataset
import evaluate
import optuna
import torch

from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import set_seed
from transformers import Trainer
from transformers import TrainingArguments


set_seed(42)

device = torch.device("cpu")

# Load dataset
train_dataset = load_dataset("imdb", split="train").shuffle().select(range(1000))
val_dataset = load_dataset("imdb", split="test").shuffle().select(range(500))
metric = evaluate.load("accuracy")

model_name = "prajjwal1/bert-tiny"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)


dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


train_dataset = dataset["train"].select(range(1000))
eval_dataset = dataset["test"].select(range(500))

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# Metric computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Training arguments
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
    per_device_train_batch_size=16,
    num_train_epochs=1,
    learning_rate=1e-6,
)

# Trainer
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# Optuna objective
def objective(trial):
    trainer.args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True)
    trainer.args.per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size", [8, 16]
    )
    trainer.args.num_train_epochs = trial.suggest_int(
        "num_train_epochs", 1, 2
    )  # <<< adjust for early stop

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_accuracy"]


# Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=3)


# Results
print("Best trial:")
print(study.best_trial)
