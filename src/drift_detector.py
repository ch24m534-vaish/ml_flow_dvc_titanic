"""
Drift Detector using PSI for both numeric and categorical features
"""

import argparse, sys, numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from collections import Counter
import yaml


# -----------------------------
# PSI Calculation
# -----------------------------
def population_stability_index(expected, actual):
    psi = 0.0
    eps = 1e-8
    for e, a in zip(expected, actual):
        e_p = max(e, eps)
        a_p = max(a, eps)
        psi += (e_p - a_p) * np.log(e_p / a_p)
    return psi

# -----------------------------
# Detect if column is numeric
# -----------------------------
def is_numeric(df, colname):
    sample = df.select(col(colname)).na.drop().limit(50).rdd.flatMap(lambda x: x).collect()
    try:
        [float(s) for s in sample]
        return True
    except:
        return False

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, help="Old (training) dataset path")
    parser.add_argument("--new", required=True, help="New incoming dataset path")
    parser.add_argument("--params", required=True, help="YAML file with configs")  #instead of --features
    args = parser.parse_args()

    #Load params.yaml
    with open(args.params) as f:
        params = yaml.safe_load(f)

    # Load configs from params.yaml
    drift_conf = params.get("drift", {})
    features = drift_conf.get("features", ["Age", "Fare", "Pclass"])
    psi_threshold = drift_conf.get("psi_threshold", 0.1)

    spark_conf = params.get("spark", {})


    spark = (SparkSession.builder
             .appName("TitanicModelTraining")
             .config("spark.executor.cores", spark_conf.get("executor_cores", 2))
             .config("spark.executor.memory", spark_conf.get("executor_memory", "4g"))
             .config("spark.executor.instances", spark_conf.get("executor_instances", 2))
             .getOrCreate()
             )
    old_df = spark.read.option("header","true").csv(args.old)
    new_df = spark.read.option("header","true").csv(args.new)

    drifted = []

    for feat in features:
        if feat not in old_df.columns or feat not in new_df.columns:
            continue

        if is_numeric(old_df, feat) and is_numeric(new_df, feat):
            # ---------------- Numeric Drift ----------------
            old_vals = old_df.select(col(feat).cast("double")).na.drop().rdd.flatMap(lambda x: x).collect()
            new_vals = new_df.select(col(feat).cast("double")).na.drop().rdd.flatMap(lambda x: x).collect()
            if not old_vals or not new_vals:
                continue

            edges = np.histogram_bin_edges(old_vals, bins=10)  # based on old data
            old_hist, _ = np.histogram(old_vals, bins=edges)
            new_hist, _ = np.histogram(new_vals, bins=edges)

            old_hist = old_hist.astype(float)/old_hist.sum()
            new_hist = new_hist.astype(float)/new_hist.sum()

            psi = population_stability_index(old_hist, new_hist)

        else:
            # ---------------- Categorical Drift ----------------
            old_vals = old_df.select(col(feat)).na.fill("NULL").rdd.flatMap(lambda x: x).collect()
            new_vals = new_df.select(col(feat)).na.fill("NULL").rdd.flatMap(lambda x: x).collect()
            if not old_vals or not new_vals:
                continue

            old_ct = Counter(old_vals)
            new_ct = Counter(new_vals)
            keys = set(old_ct.keys()) | set(new_ct.keys())

            old_list = [old_ct.get(k, 0)/len(old_vals) for k in keys]
            new_list = [new_ct.get(k, 0)/len(new_vals) for k in keys]

            psi = population_stability_index(old_list, new_list)

        # ---------------- Drift Decision ----------------
        if psi >= psi_threshold:
            drifted.append((feat, psi))

    # ---------------- Final Result ----------------
    if drifted:
        print(" Drift detected:", drifted)
        code = 2
    else:
        print("No significant drift.")
        code = 0

    spark.stop()
    sys.exit(code)

if __name__ == "__main__":
    main()
