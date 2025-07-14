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
    if SparkSession is None:
        print("This example requires pyspark. Please install it to run this script.")
    else:
        spark = SparkSession.builder.appName("OptunaSparkExample").getOrCreate()
        study = optuna.create_study()

        for i in range(20):
            trial = study.ask()
            x = trial.suggest_float("x", -10, 10)
            rdd = spark.sparkContext.parallelize([x])
            result = rdd.map(evaluate).collect()[0]
            study.tell(trial, result)
            print(f"Trial {i + 1}: x = {x:.4f}, result = {result:.4f}")

        print("\nBest trial:")
        print(f"  Value: {study.best_value}")
        print(f"  Params: {study.best_trial.params}")
        spark.stop()
