"""
Spark with Ask-and-Tell

An example showing how to use Optuna's ask-and-tell interface with Apache Spark
to distribute the evaluation of trials.

This script uses PySpark's RDD to parallelize a simple quadratic function.
"""

import optuna
from pyspark.sql import SparkSession


def evaluate(trial_number, x):
    # Simulate a trial evaluation on a Spark worker node
    # Keep the trial_number with the result as Spark does not necessarily return the results in
    # order
    return trial_number, (x - 2) ** 2


if __name__ == "__main__":
    spark = SparkSession.builder.appName("OptunaSparkExample").getOrCreate()
    study = optuna.create_study()

    n_batches = 10
    batch_size = 30

    for batch in range(n_batches):
        trials = [study.ask() for _ in range(batch_size)]
        params = [[t.number, t.suggest_float("x", -10, 10)] for t in trials]
        rdd = spark.sparkContext.parallelize(params, len(params))
        results = rdd.map(lambda x: evaluate(*x)).collect()
        for trial_number, result in results:
            study.tell(trial_number, result)
            print(f"Trial#{trial_number}: {result=:.4e}")

    print("\nBest trial:")
    print(f"\tValue: {study.best_value}")
    print(f"\tParams: {study.best_trial.params}")
    print(f"\tNumber of evaluations: {len(study.trials)}")
    spark.stop()
