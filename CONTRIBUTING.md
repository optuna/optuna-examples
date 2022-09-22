# Contribution Guidelines

It’s such an honor to have you on board!

If you feel like giving your hand to us, here are some ways
- Add an example
    - If you have an idea for an example, please feel free to open a PR as draft to discuss design or work on your example.
- Report a bug
    - If you find some bug, don't hesitate to report it! Your reports matter.

If you choose to write some code, we have some conventions as follows.

- [Guidelines](#guidelines)
- [Continuous Integration and Local Verification](#continuous-integration-and-local-verification)
- [Creating a Pull Request](#creating-a-pull-request)

## Guidelines

### Setup Optuna

See the [optuna/optuna/CONTRIBUTING.MD](https://github.com/optuna/optuna/blob/master/CONTRIBUTING.md) file to see how to install Optuna.

### Checking the Format and Coding Style

Code is formatted with [black](https://github.com/psf/black),
Coding style is checked with [flake8](http://flake8.pycqa.org) and [isort](https://pycqa.github.io/isort/)
and additional conventions are described in the [Wiki](https://github.com/optuna/optuna/wiki/Coding-Style-Conventions).

If your environment is missing some dependencies such as black, flake8, or isort,
you will be asked to install them.

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

## Creating a Pull Request

When you are ready to create a pull request, please try to keep the following in mind.

### Title

The title of your pull request should

- briefly describe and reflect the changes
- wrap any code with backticks
- not end with a period

#### Example

Add new example for using Optuna to tune GPT-4

### Description

The description of your pull request should

- describe the motivation
- describe the changes
- if still work-in-progress, describe remaining tasks
