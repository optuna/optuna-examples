"""
Optuna example for fine-tuning a BERT-based text classification model on the IMDb dataset
with hyperparameter optimization using Optuna. In this example, we fine-tune a lightweight
pre-trained BERT model on a small subset of the IMDb dataset to classify movie reviews as
positive or negative. We optimize the validation accuracy by tuning the learning rate
and batch size.To learn more, you can check the following
documentation: https://huggingface.co/docs/transformers/en/hpo_train?
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset = load_dataset("imdb", split="train").shuffle(seed=42).select(range(1000))
valid_dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(500))

model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)


tokenized_train = train_dataset.map(tokenize, batched=True).select_columns(["input_ids", "label"])
tokenized_valid = valid_dataset.map(tokenize, batched=True).select_columns(["input_ids", "label"])


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
    eval_strategy="epoch",
    save_strategy="best",
    load_best_model_at_end=True,
    logging_strategy="epoch",
    report_to="none",
)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64, 128]
        ),
    }


best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=5,
    compute_objective=compute_objective,
)

print(best_run)
