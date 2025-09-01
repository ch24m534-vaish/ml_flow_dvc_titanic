from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
import mlflow.pyfunc
import pandas as pd
import os
from typing import Optional
import uvicorn



# --------------------------
# MLflow setup
# --------------------------
mlflow.set_tracking_uri("http://10.255.255.254:5000")


# --------------------------
# FastAPI app
# --------------------------
app = FastAPI()

# --------------------------
# Start Spark
# --------------------------
spark = SparkSession.builder.appName("TitanicAPI").getOrCreate()

# --------------------------
# Load preprocessing pipeline
# --------------------------
pipeline_model_path = "data/processed/train_parquet/preprocess_pipeline"
if not os.path.exists(pipeline_model_path):
    raise RuntimeError(f"❌ Pipeline path not found: {pipeline_model_path}")
pipeline_model = PipelineModel.load(pipeline_model_path)

# --------------------------
# Load best MLflow model
# --------------------------
mlflow_model_uri = "models:/Titanic_RandomForest/Production"
mlflow_model = mlflow.pyfunc.load_model(mlflow_model_uri)

# --------------------------
# Schema for Spark input
# --------------------------
passenger_schema = StructType([
    StructField("Pclass", IntegerType(), True),
    StructField("Sex", StringType(), True),
    StructField("Age", DoubleType(), True),      # Nullable float
    StructField("SibSp", IntegerType(), True),
    StructField("Parch", IntegerType(), True),
    StructField("Fare", DoubleType(), True),
    StructField("Embarked", StringType(), True),
])

# --------------------------
# Request schema (Pydantic)
# --------------------------
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: Optional[float] = None
    SibSp: int
    Parch: int
    Fare: Optional[float] = None
    Embarked: str

# --------------------------
# API Endpoint
# --------------------------
@app.post("/predict")
def predict(passenger: Passenger):
    # Convert request to dict
    input_dict = passenger.dict()

    # Create Spark DF with explicit schema
    input_df = spark.createDataFrame([input_dict], schema=passenger_schema)

    # Apply preprocessing pipeline
    processed_df = pipeline_model.transform(input_df)

    # Convert processed features into Pandas for MLflow
    processed_df = processed_df.toPandas()

    # Predict with MLflow model
    preds = mlflow_model.predict(processed_df)

    return {"prediction": int(preds[0])}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )