"""
Visualizing Optimization Timeline with plot_timeline.

This example demonstrates how to use Optuna's plot_timeline function to visualize
the progression of trials over time. The timeline plot shows when each trial was
executed and their corresponding objective values, helping to understand the
optimization process dynamics.

This is particularly useful for:
- Monitoring long-running optimizations
- Identifying time-based patterns in performance
- Debugging parallel optimization processes
- Understanding convergence behavior over time
"""

import optuna
from optuna.visualization import plot_timeline
import time
import random


def objective(trial):
    """Objective function with variable execution time to demonstrate timeline.

    This function simulates a machine learning training process where:
    1. Some parameter combinations take longer to evaluate
    2. Performance varies based on hyperparameters
    3. There's realistic variance in execution time

    Args:
        trial: Optuna trial object for suggesting hyperparameters.

    Returns:
        float: Objective value to minimize (simulated validation error).
    """
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)

    # Simulate variable training time based on parameters
    # More estimators and deeper trees take longer
    base_time = 0.5
    complexity_factor = (n_estimators / 100) * (max_depth / 10)
    execution_time = base_time + complexity_factor * 2 + random.uniform(0, 1)

    # Simulate training time
    time.sleep(execution_time)

    # Simulate model performance (lower is better)
    # Optimal values: n_estimators~100, max_depth~8, learning_rate~0.1, subsample~0.8
    performance = (
        abs(n_estimators - 100) * 0.001
        + abs(max_depth - 8) * 0.01
        + abs(learning_rate - 0.1) * 2
        + abs(subsample - 0.8) * 0.5
        + random.gauss(0, 0.05)  # Add noise
    )

    return performance


def create_sample_study_with_timeline():
    """Create a study with trials that have realistic timing patterns."""
    print("Creating sample study for timeline demonstration...")

    study = optuna.create_study(
        direction="minimize",
        study_name="timeline_example",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Run optimization with progress reporting
    def callback(study, trial):
        """Callback to show progress during optimization."""
        if trial.number % 5 == 0:
            print(
                f"Trial {trial.number}: value = {trial.value:.4f}, "
                f"duration = {trial.duration.total_seconds():.2f}s"
            )

    print("Running optimization trials...")
    print("Note: Each trial has variable execution time to demonstrate timeline visualization")

    study.optimize(objective, n_trials=30, callbacks=[callback])

    return study


def analyze_timeline_patterns(study):
    """Analyze and report timeline patterns."""
    print("\n" + "=" * 60)
    print("TIMELINE ANALYSIS:")
    print("=" * 60)

    trials = study.trials
    durations = [t.duration.total_seconds() for t in trials if t.duration]
    values = [t.value for t in trials if t.value is not None]

    if durations and values:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        best_trial = study.best_trial
        best_duration = best_trial.duration.total_seconds() if best_trial.duration else "N/A"

        print(f"Total trials: {len(trials)}")
        print(f"Average trial duration: {avg_duration:.2f} seconds")
        print(f"Fastest trial: {min_duration:.2f} seconds")
        print(f"Slowest trial: {max_duration:.2f} seconds")
        print(f"Best trial duration: {best_duration} seconds")
        print(f"Best objective value: {study.best_value:.4f}")

        # Find correlation between duration and performance
        completed_trials = [
            (t.duration.total_seconds(), t.value)
            for t in trials
            if t.duration and t.value is not None
        ]

        if len(completed_trials) > 5:
            durations, values = zip(*completed_trials)
            correlation = calculate_correlation(durations, values)
            print(f"Duration-Performance correlation: {correlation:.3f}")

            if correlation > 0.3:
                print("→ Longer trials tend to perform worse")
            elif correlation < -0.3:
                print("→ Longer trials tend to perform better")
            else:
                print("→ No strong correlation between duration and performance")


def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))

    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    return numerator / denominator if denominator != 0 else 0


def main():
    """Main function to demonstrate timeline visualization."""
    print("Optuna Timeline Visualization Example")
    print("=" * 40)

    # Create study with timing data
    study = create_sample_study_with_timeline()

    # Analyze timeline patterns
    analyze_timeline_patterns(study)

    print("\nGenerating timeline visualization...")

    # Create timeline plot
    fig = plot_timeline(study)

    # Customize the plot
    fig.update_layout(
        title="Optimization Timeline - Trial Execution Over Time",
        title_x=0.5,
        width=1000,
        height=600,
        xaxis_title="Timeline",
        yaxis_title="Objective Value",
    )

    # Save the plot
    fig.write_html("optimization_timeline.html")
    print("Timeline plot saved as 'optimization_timeline.html'")

    # Show plot if possible
    try:
        fig.show()
    except Exception:
        print("Plot display not available in this environment.")
        print("Open 'optimization_timeline.html' in your browser to view the plot.")

    # Print interpretation guide
    print("\n" + "=" * 60)
    print("HOW TO INTERPRET THE TIMELINE PLOT:")
    print("=" * 60)
    print("• Each point represents a completed trial")
    print("• X-axis shows when trials were executed (timeline)")
    print("• Y-axis shows the objective value achieved")
    print("• Color/marker may indicate trial performance ranking")
    print("• Gaps in timeline may indicate longer-running trials")
    print("• Pattern analysis helps optimize the optimization process itself")
    print("• Use this to:")
    print("  - Monitor progress in real-time")
    print("  - Identify bottlenecks in parallel optimization")
    print("  - Understand convergence patterns")
    print("  - Debug timing issues in distributed setups")


if __name__ == "__main__":
    main()
