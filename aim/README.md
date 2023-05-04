# Tracking optimization process with [`aim`](https://aimstack.io/)

![aimui]()

Optuna example that optimizes a classifier configuration for the Iris dataset using
scikit-learn and records hyperparameters and metrics using aim.

In this example we optimize random forest classifier for the Iris dataset. All
hyperparameters and metrics will be logged to `.aim` directory via integration callback.

You can run this example as follows:

```sh
$ python aim_integration.py
```

After the script finishes, run the aim UI:

```sh
$ aim up
```

and view the optimization results at http://127.0.0.1:43800.

See also the [official example provided by aim](https://github.com/aimhubio/aim/blob/main/examples/optuna_track.py).
