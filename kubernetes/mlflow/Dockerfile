# NOTE(crcrpar): optuna/optuna:py3.7-dev can be used as base instead.
FROM python:3.11-slim

WORKDIR /usr/src/

RUN pip install --no-cache-dir optuna psycopg2-binary mlflow \
    && pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install pytorch-lightning

COPY pytorch_lightning_distributed.py .
COPY check_study.sh .
