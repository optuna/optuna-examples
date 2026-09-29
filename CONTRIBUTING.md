# Contribution Guidelines

Thank you for your interest in contributing to Optuna Examples!

> [!NOTE]
> Optuna Examples is temporarily pausing external pull request submissions so that maintainers can focus on maintaining the existing examples within our available review capacity.
> In the meantime, bug reports, feedback, and suggestions remain welcome through [GitHub Issues](https://github.com/optuna/optuna-examples/issues).
The following guidelines are provided for maintainers and anyone working on a local copy of the repository.

- [Guidelines](#guidelines)
- [Continuous Integration and Local Verification](#continuous-integration-and-local-verification)

## Guidelines

### Setup Optuna

See the [optuna/optuna/CONTRIBUTING.MD](https://github.com/optuna/optuna/blob/master/CONTRIBUTING.md) file to see how to install Optuna.

### Checking the Format and Coding Style

Code is formatted with [black](https://github.com/psf/black),
Coding style is checked with [flake8](http://flake8.pycqa.org) and [isort](https://pycqa.github.io/isort/)
and additional conventions are described in the [Wiki](https://github.com/optuna/optuna/wiki/Coding-Style-Conventions).

If your environment is missing some dependencies such as black, flake8, or isort,
you will be asked to install them.

You can use `pre-commit` to automatically check the format, coding style, and type hints before committing.
The following commands automatically fix format errors by auto-formatters.

```bash
# Install `pre-commit`.
$ pip install pre-commit

$ pre-commit install
$ pre-commit run --all-files
```

## Continuous Integration and Local Verification

This repository uses GitHub Actions.

### Local Verification

By installing [`act`](https://github.com/nektos/act#installation) and Docker, you can run
tests written for GitHub Actions locally.

```bash
JOB_NAME=checks
act -j $JOB_NAME
```

Currently, you can run the following jobs:

- `checks`
  - Checks the format
- `examples`
  - Run the examples

To run a specific example job:

```bash
act -j examples -W path/to/example.yml/file
```

Usually, the example.yml file will be in the [`.github/workflows/`](.github/workflows/) directory.
