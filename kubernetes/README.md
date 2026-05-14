# Distributed Optimization on Kubernetes

This folder contains two kinds of examples with Kubernetes: one is based on [`sklearn_simple.py`](../sklearn/sklearn_simple.py).

Currently, [`simple/sklearn_distributed.py`](./simple/sklearn_distributed.py) uses POSTGRESQL for their backend of `optuna.Study.optimize` to be parallelized.

