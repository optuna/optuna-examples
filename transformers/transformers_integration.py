"""
Optuna example that optimizes Transformer-based models using Hugging Face Transformers.

In this example, we optimize the validation accuracy for sentiment classification on the IMDb
dataset using DistilBERT. We optimize hyperparameters such as the learning rate, batch size,
and number of epochs.It is recommended to run this script on a GPU for faster experimentation.
As it is too time consuming to use the full IMDb dataset during
hyperparameter search, we use a small subset of it for experimentation.
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


# Set seed for reproducibility
set_seed(42)

# Load IMDb dataset
dataset = load_dataset("imdb")
metric = evaluate.load("accuracy")  # Replaces deprecated load_metric

# Model name
model_name = "lvwerra/distilbert-imdb"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ➔ Define train and eval datasets here (before slicing)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Slice datasets for faster experiments
train_dataset = train_dataset.select(range(1500))
eval_dataset = eval_dataset.select(range(500))

# Model config
config = AutoConfig.from_pretrained(model_name, num_labels=2)


# Model initialization function
def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )


# Compute accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Define Optuna search space
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16]
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 3),
    }


# Training arguments
training_args = TrainingArguments(
    logging_strategy="epoch",
    output_dir="./results",
    eval_strategy="epoch",  # eval_strategy ➔ evaluation_strategy
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="none",
    fp16=False,
    dataloader_pin_memory=torch.cuda.is_available(),  # Safer on CPU or smaller GPUs
)

# Initialize Trainer
trainer = Trainer(
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    model_init=model_init,
    compute_metrics=compute_metrics,
)


# Define objective function for Optuna optimization
def objective(trial):
    # Update hyperparameters
    training_args.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    training_args.per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size", [8, 16]
    )
    training_args.num_train_epochs = trial.suggest_int("num_train_epochs", 2, 3)

    trainer.args = training_args

    # Perform training
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    return eval_results["eval_accuracy"]


# Run Optuna hyperparameter search
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

# Get best trial
best_trial = study.best_trial

print("Best trial found:")
print(best_trial)
