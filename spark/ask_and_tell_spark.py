"""
Spark with Ask-and-Tell

An example showing how to use Optuna's ask-and-tell interface with Apache Spark
to distribute the evaluation of trials.

This script uses PySpark's RDD to parallelize a simple quadratic function.
"""

import optuna
from pyspark.sql import SparkSession


def evaluate(x):
    return (x - 2) ** 2  # Simulate a trial evaluation on a Spark worker node


if __name__ == "__main__":
    spark = SparkSession.builder.appName("OptunaSparkExample").getOrCreate()
    study = optuna.create_study()

    for i in range(20):
        trial = study.ask()
        x = trial.suggest_float("x", -10, 10)
        rdd = spark.sparkContext.parallelize([x])
        result = rdd.map(evaluate).collect()[0]
        study.tell(trial, result)
        print(f"Trial#{trial.number}: {x=:.4e}, {result=:.4e}")

    print("\nBest trial:")
    print(f"\tValue: {study.best_value}")
    print(f"\tParams: {study.best_trial.params}")
    spark.stop()
