# How to Run Preferential Optimization Image

First, ensure the necessary packages are installed by executing the following command in your terminal:

```bash
$ pip install "optuna>=3.3.0" "optuna-dashboard[preferential]>=0.13.0b1" pillow
```

Next, execute the Python script.

```bash
$ python generator.py
```

Then, launch Optuna Dashboard in a separate process using the following command.

```bash
optuna-dashboard sqlite:///example.db --artifact-dir ./artifact
```