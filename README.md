Optuna Examples
================

This page contains a list of example codes written with Optuna.

<details open>
<summary>Simplest Codeblock</summary>

```python
import optuna


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    return x ** 2


if __name__ == "__main__":
    study = optuna.create_study()
    # The optimization finishes after evaluating 1000 times or 3 seconds.
    study.optimize(objective, n_trials=1000, timeout=3)
    print(f"Best params is {study.best_params} with value {study.best_value}")
```
</details>

> [!NOTE]
> If you are interested in a quick start of [Optuna Dashboard](https://github.com/optuna/optuna-dashboard) with in-memory storage, please take a look at [this example](./dashboard/run_server_simple.py).

> [!TIP]
> Couldn't find your usecase?
> [FAQ](https://optuna.readthedocs.io/en/stable/faq.html) might be helpful for you to implement what you want.
> In this example repository, you can also find the examples for the following scenarios:
> 1. [Objective function with additional arguments](./sklearn/sklearn_additional_args.py), which is useful when you would like to pass arguments besides `trial` to your objective function.
>
> 2. [Manually provide trials with sampler](./faq/enqueue_trial.py), which is useful when you would like to force certain parameters to be sampled.
>
> 3. [Callback to control the termination criterion of study](./faq/max_trials_callback.py), which is useful when you would like to define your own termination criterion other than `n_trials` or `timeout`.

## Examples for Diverse Problem Setups

Here are the URLs to the example codeblocks to the corresponding setups.

<details open>
<summary>Simple Black-box Optimization</summary>

* [Quadratic Function](./basic/quadratic.py)
* [Quadratic Multi-Objective Function](./basic/quadratic_multi_objective.py)
* [Quadratic Function with Constraints](./basic/quadratic_constraint.py)
</details>

<details open>
<summary>Multi-Objective Optimization</summary>

* [Optimization with BoTorch](./multi_objective/botorch_simple.py)
* [Optimization of Multi-Layer Perceptron with PyTorch](./multi_objective/pytorch_simple.py)
</details>

<details open>
<summary>Machine Learning (Incl. LightGBMTuner and OptunaSearchCV)</summary>

* [AllenNLP](./allennlp/allennlp_simple.py)
* [AllenNLP (Jsonnet)](./allennlp/allennlp_jsonnet.py)
* [Catalyst](./pytorch/catalyst_simple.py)
* [CatBoost](./catboost/catboost_simple.py)
* [Chainer](./chainer/chainer_simple.py)
* [ChainerMN](./chainer/chainermn_simple.py)
* [Dask-ML](./dask_ml/dask_ml_simple.py)
* [FastAI](./fastai/fastai_simple.py)
* [Haiku](./haiku/haiku_simple.py)
* [Keras](./keras/keras_simple.py)
* [LightGBM](./lightgbm/lightgbm_simple.py)
* [LightGBM Tuner](./lightgbm/lightgbm_tuner_simple.py)
* [PyTorch](./pytorch/pytorch_simple.py)
* [PyTorch Ignite](./pytorch/pytorch_ignite_simple.py)
* [PyTorch Lightning](./pytorch/pytorch_lightning_simple.py)
* [PyTorch Lightning (DDP)](./pytorch/pytorch_lightning_ddp.py)
* [RAPIDS](./rapids/rapids_simple.py)
* [Scikit-learn](./sklearn/sklearn_simple.py)
* [Scikit-learn OptunaSearchCV](./sklearn/sklearn_optuna_search_cv_simple.py)
* [Scikit-image](./skimage/skimage_lbp_simple.py)
* [SKORCH](./pytorch/skorch_simple.py)
* [Tensorflow](./tensorflow/tensorflow_estimator_simple.py)
* [Tensorflow (eager)](./tensorflow/tensorflow_eager_simple.py)
* [XGBoost](./xgboost/xgboost_simple.py)

If you are looking for an example of reinforcement learning, please take a look at the following:
* [Optimization of Hyperparameters for Stable-Baslines Agent](./rl/sb3_simple.py)

</details>

<details open>
<summary>Pruning</summary>

The following example demonstrates how to implement pruning logic with Optuna.

* [Simple pruning (scikit-learn)](./basic/pruning.py)

In addition, integration modules are available for the following libraries, providing simpler interfaces to utilize pruning.

* [Pruning with Catalyst Integration Module](./pytorch/catalyst_simple.py)
* [Pruning with CatBoost Integration Module](./catboost/catboost_pruning.py)
* [Pruning with Chainer Integration Module](./chainer/chainer_integration.py)
* [Pruning with ChainerMN Integration Module](./chainer/chainermn_integration.py)
* [Pruning with FastAI Integration Module](./fastai/fastai_simple.py)
* [Pruning with Keras Integration Module](./keras/keras_integration.py)
* [Pruning with LightGBM Integration Module](./lightgbm/lightgbm_integration.py)
* [Pruning with PyTorch Integration Module](./pytorch/pytorch_simple.py)
* [Pruning with PyTorch Ignite Integration Module](./pytorch/pytorch_ignite_simple.py)
* [Pruning with PyTorch Lightning Integration Module](./pytorch/pytorch_lightning_simple.py)
* [Pruning with PyTorch Lightning Integration Module (DDP)](./pytorch/pytorch_lightning_ddp.py)
* [Pruning with Tensorflow Integration Module](./tensorflow/tensorflow_estimator_integration.py)
* [Pruning with XGBoost Integration Module](./xgboost/xgboost_integration.py)
* [Pruning with XGBoost Integration Module (Cross Validation Version)](./xgboost/xgboost_cv_integration.py)
</details>

<details open>
<summary>Samplers</summary>

* [Warm Starting CMA-ES](./samplers/warm_starting_cma.py)

If you are interested in defining a user-defined sampler, here is an example:
* [SimulatedAnnealingSampler](./samplers/simulated_annealing_sampler.py)
</details>

<details open>
<summary>Terminator</summary>

* [Optuna Terminator](./terminator/terminator_simple.py)
* [OptunaSearchCV with Terminator](./terminator/terminator_search_cv.py)
</details>

<details open>
<summary>Visualization</summary>

* [Visualizing Study](https://colab.research.google.com/github/optuna/optuna-examples/blob/main/visualization/plot_study.ipynb)
* [Visualizing Study with HiPlot](https://colab.research.google.com/github/optuna/optuna-examples/blob/main/hiplot/plot_study.ipynb)
</details>

<details open>
<summary>Distributed Optimization</summary>

* [Optimizing on Dask Cluster](./dask/dask_simple.py)
* [Optimizing on Kubernetes](./kubernetes/README.md)
* [Optimizing with Ray's Joblib Backend](./ray/ray_joblib.py)
</details>

<details open>
<summary>MLOps Platform</summary>

* [Tracking Optimization Process with aim](./aim/aim_integration.py)
* [Tracking Optimization Process with MLflow](./mlflow/keras_mlflow.py)
* [Tracking Optimization Process with Weights & Biases](./wandb/wandb_integration.py)
* [Optimization with Hydra](./hydra/simple.py)
</details>

<details open>
<summary>External Projects Using Optuna</summary>

* [Hugging Face Trainer's Hyperparameter Search](https://huggingface.co/docs/transformers/main/main_classes/trainer#transformers.Trainer.hyperparameter_search)
* [Allegro Trains](https://github.com/allegroai/trains)
* [BBO-Rietveld: Automated Crystal Structure Refinement](https://github.com/quantumbeam/BBO-Rietveld)
* [Catalyst](https://github.com/catalyst-team/catalyst)
* [CuPy](https://github.com/cupy/cupy)
* [Hydra's Optuna Sweeper Plugin](https://hydra.cc/docs/next/plugins/optuna_sweeper/)
* [Mozilla Voice STT](https://github.com/mozilla/DeepSpeech)
* [neptune.ai](https://neptune.ai)
* [OptGBM: A scikit-learn Compatible LightGBM Estimator with Optuna](https://github.com/Y-oHr-N/OptGBM)
* [Optuna-distributed](https://github.com/xadrianzetx/optuna-distributed)
* [PyKEEN](https://github.com/pykeen/pykeen)
* [RL Baselines Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
* [Hyperparameter Optimization for Machine Learning, Code Repository for Online Course](https://github.com/solegalli/hyperparameter-optimization)
* [Property-guided molecular optimization using MolMIM with CMA-ES](https://github.com/olachinkei/BioNeMo_WandB/blob/main/Molecule/03_Molecule_LLM.ipynb)
</details>

> [!IMPORTANT]
> PRs to add additional real-world examples or projects are welcome!

### Running with Optuna's Docker images?

Our Docker images for most examples are available with the tag ending with `-dev`.
For example, [PyTorch Simple](./pytorch/pytorch_simple.py) can be run via:

```bash
$ docker run --rm -v $(pwd):/prj -w /prj optuna/optuna:py3.11-dev python pytorch/pytorch_simple.py
```

Additionally, our visualization example can also be run on Jupyter Notebook by opening `localhost:8888` in your browser after executing the following:

```bash
$ docker run -p 8888:8888 --rm optuna/optuna:py3.11-dev jupyter notebook --allow-root --no-browser --port 8888 --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''
```
