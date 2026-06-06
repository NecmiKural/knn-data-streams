"""
COMPX523-26A Assignment 1 - Experiment Runner
Student ID: 1689664

Runs all 24 experiments (3 algorithms x 2 datasets x 4 configs)
and generates the CSV result file.

Usage: python run_experiments.py
"""

import os
import csv
import time

# Adjust JAVA_HOME if needed (uncomment and modify for your system)
# os.environ['JAVA_HOME'] = '/opt/homebrew/opt/openjdk@11'

# Fix Turkish locale issue: MOA's ARFF parser uses case-insensitive string
# matching which breaks under tr_TR locale (I/i -> İ/ı rules).
# Must be set BEFORE capymoa starts the JVM.
import jpype
if not jpype.isJVMStarted():
    os.environ.setdefault('JAVA_TOOL_OPTIONS', '-Duser.language=en -Duser.country=US')

from capymoa.evaluation import prequential_evaluation
from capymoa.datasets import Electricity, RBFm_100k
from knnAssignment import KNNBase, KNN_cw_1689664, KNN_asw_1689664

STUDENT_ID = "1689664"
OUTPUT_FILE = "1689664_results.csv"
MAX_INSTANCES = 50000  # use first 50k instances for both datasets

# Parameter configurations
CONFIGS = [
    {"k": 3,  "w": 500},
    {"k": 21, "w": 500},
    {"k": 3,  "w": 100},
    {"k": 21, "w": 100},
]

# Dataset definitions
DATASETS = {
    "electricity": Electricity,
    "rbf": RBFm_100k,
}

# Algorithm definitions
# Note: KNNBase uses 'window_size' param, our classes use 'w'
ALGORITHMS = {
    "knn": {
        "class": KNNBase,
        "window_param": "window_size",  # KNNBase uses 'window_size'
    },
    "knn-cw": {
        "class": KNN_cw_1689664,
        "window_param": "w",
    },
    "knn-asw": {
        "class": KNN_asw_1689664,
        "window_param": "w",
    },
}


def run_experiment(algo_name, algo_info, dataset_name, dataset_class, config):
    """Run a single experiment and return the result row."""
    k = config["k"]
    w = config["w"]

    # Create fresh stream
    stream = dataset_class()
    schema = stream.get_schema()

    # Create classifier with correct parameter name
    params = {"schema": schema, "k": k, algo_info["window_param"]: w}
    learner = algo_info["class"](**params)

    # Run prequential evaluation
    start_time = time.time()
    result = prequential_evaluation(
        stream=stream,
        learner=learner,
        max_instances=MAX_INSTANCES,
    )
    wallclock = result.wallclock()

    # Extract metrics
    accuracy = round(result.cumulative.accuracy(), 3)
    recall = round(result.cumulative.recall(), 3)
    precision = round(result.cumulative.precision(), 3)
    wallclock = round(wallclock, 3)

    row = {
        "student_id": STUDENT_ID,
        "dataset": dataset_name,
        "knn_identifier": algo_name,
        "window_size": w,
        "k": k,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "wallclock": wallclock,
    }

    print(f"  {algo_name} | {dataset_name} | k={k}, w={w} | "
          f"acc={accuracy}, rec={recall}, prec={precision}, time={wallclock}s")

    return row


def main():
    print(f"=== COMPX523-26A Assignment 1 - Experiment Runner ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Max instances: {MAX_INSTANCES}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    results = []
    total = len(ALGORITHMS) * len(DATASETS) * len(CONFIGS)
    count = 0

    for algo_name, algo_info in ALGORITHMS.items():
        for dataset_name, dataset_class in DATASETS.items():
            for config in CONFIGS:
                count += 1
                print(f"[{count}/{total}]", end=" ")
                row = run_experiment(
                    algo_name, algo_info,
                    dataset_name, dataset_class,
                    config
                )
                results.append(row)
                print()

    # Write CSV
    fieldnames = [
        "student_id", "dataset", "knn_identifier", "window_size",
        "k", "accuracy", "recall", "precision", "wallclock"
    ]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== Done! Results written to {OUTPUT_FILE} ===")
    print(f"Total experiments: {len(results)}")


if __name__ == "__main__":
    main()
