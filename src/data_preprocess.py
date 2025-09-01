from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, first
from pyspark.sql.types import IntegerType, DoubleType
import argparse
import os
import yaml

def preprocess_data(input_path: str, params_path, output_path: str, save_pipeline=True):
    with open(params_path) as f:
        params = yaml.safe_load(f)
    spark_conf = params.get("spark", {})

    spark = (SparkSession.builder
             .appName("TitanicPreprocessing")
             .config("spark.executor.cores", spark_conf.get("executor_cores", 2))
             .config("spark.executor.memory", spark_conf.get("executor_memory", "4g"))
             .config("spark.executor.instances", spark_conf.get("executor_instances", 2))
             .getOrCreate()
             )

    # Read raw CSV
    raw_df = spark.read.option("header", "true").csv(input_path)

    # Drop rows with missing label
    clean_df = raw_df.na.drop(subset=["Survived"])

    # Cast numeric columns
    clean_df = clean_df\
        .withColumn("Pclass", col("Pclass").cast(IntegerType()))\
        .withColumn("Age", col("Age").cast(DoubleType()))\
        .withColumn("SibSp", col("SibSp").cast(IntegerType()))\
        .withColumn("Parch", col("Parch").cast(IntegerType()))\
        .withColumn("Fare", col("Fare").cast(DoubleType()))

    # -------------------------
    # Impute Embarked with mode
    # -------------------------
    mode_row = clean_df.groupBy("Embarked").count().orderBy(col("count").desc()).first()
    mode_embarked = mode_row["Embarked"] if mode_row else "S"
    clean_df = clean_df.fillna({"Embarked": mode_embarked})

    # Imputer for numeric columns
    imputer_age = Imputer(inputCols=["Age"], outputCols=["Age"])
    imputer_fare = Imputer(inputCols=["Fare"], outputCols=["Fare"])

    # Encode categorical columns
    indexers = [
        StringIndexer(inputCol="Sex", outputCol="SexIndex", handleInvalid="keep"),
        StringIndexer(inputCol="Embarked", outputCol="EmbarkedIndex", handleInvalid="keep")
    ]

    # Assemble features
    feature_columns = ["Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "EmbarkedIndex"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Pipeline
    pipeline = Pipeline(stages=indexers + [imputer_age, imputer_fare, assembler])
    pipeline_model = pipeline.fit(clean_df)
    processed_df = pipeline_model.transform(clean_df)

    # Final columns
    final_df = processed_df.select(
        "features", *feature_columns, col("Survived").cast("int").alias("label")
    )

    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    final_df.write.mode("overwrite").parquet(os.path.join(output_path, "data"))

    # Save pipeline for reuse
    if save_pipeline:
        pipeline_model.save(os.path.join(output_path, "preprocess_pipeline"))

    print(f"Processed data saved at {os.path.join(output_path, 'data')}")
    print(f"Pipeline saved at {os.path.join(output_path, 'preprocess_pipeline')}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw Titanic CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed parquet folder")
    parser.add_argument("--params", required=True, help="Path to params.yaml")
    args = parser.parse_args()
    preprocess_data(args.input, args.params, args.output)