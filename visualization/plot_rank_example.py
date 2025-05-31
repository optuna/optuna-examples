"""
Visualizing Parameter Ranking with plot_rank.

This example demonstrates how to use Optuna's plot_rank function to visualize
the ranking of parameter importance across trials. The plot_rank function helps
identify which parameters have the most significant impact on the objective values.

In this example, we'll optimize a simple quadratic function with multiple parameters
and then visualize how different parameter values rank in terms of their contribution
to the objective function performance.
"""

import optuna
from optuna.visualization import plot_rank


def objective(trial):
    """Objective function for hyperparameter optimization.

    This function simulates a machine learning model where some parameters
    are more important than others for achieving good performance.

    Args:
        trial: Optuna trial object for suggesting hyperparameters.

    Returns:
        float: Objective value to minimize.
    """
    # Important parameters (high impact on objective)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    # Moderately important parameters
    n_layers = trial.suggest_int("n_layers", 1, 5)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    # Less important parameters
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # Simulate model performance with different parameter importance
    # Learning rate and batch size have higher impact
    lr_penalty = (learning_rate - 0.01) ** 2 * 1000
    batch_penalty = (batch_size - 64) ** 2 * 0.01

    # Other parameters have moderate to low impact
    layer_penalty = (n_layers - 3) ** 2 * 10
    dropout_penalty = (dropout_rate - 0.2) ** 2 * 50
    optimizer_penalty = {"adam": 0, "sgd": 20, "rmsprop": 10}[optimizer]
    weight_decay_penalty = (weight_decay - 1e-4) ** 2 * 100

    # Add some random noise to simulate real-world variance
    import random

    noise = random.gauss(0, 5)

    return (
        lr_penalty
        + batch_penalty
        + layer_penalty
        + dropout_penalty
        + optimizer_penalty
        + weight_decay_penalty
        + noise
    )


def main():
    """Main function to run optimization and create ranking visualization."""
    print("Running hyperparameter optimization...")

    # Create study
    study = optuna.create_study(
        direction="minimize",
        study_name="parameter_ranking_example",
        sampler=optuna.samplers.TPESampler(seed=42),  # For reproducible results
    )

    # Run optimization
    study.optimize(objective, n_trials=100)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\nGenerating parameter ranking visualization...")

    # Create ranking plot
    fig = plot_rank(study)

    # Customize the plot
    fig.update_layout(
        title="Parameter Ranking - Optuna Hyperparameter Optimization",
        title_x=0.5,
        width=800,
        height=600,
    )

    # Save the plot
    fig.write_html("parameter_ranking.html")
    print("Ranking plot saved as 'parameter_ranking.html'")

    # Show plot if running in an interactive environment
    try:
        fig.show()
    except Exception:
        print("Plot display not available in this environment.")
        print("Open 'parameter_ranking.html' in your browser to view the plot.")

    # Print interpretation guide
    print("\n" + "=" * 60)
    print("HOW TO INTERPRET THE RANKING PLOT:")
    print("=" * 60)
    print("• The plot shows how parameter values correlate with objective performance")
    print("• Parameters with steeper slopes have higher importance")
    print("• Flat lines indicate parameters with little impact on the objective")
    print("• The ranking helps identify which parameters to focus on for tuning")
    print("• In this example, learning_rate and batch_size should show high importance")


if __name__ == "__main__":
    main()
