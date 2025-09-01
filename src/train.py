import time
import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score,log_loss
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
import psutil
from sklearn.metrics import roc_curve, roc_auc_score




# -------------------------------
#  Model Registration Utilities
# -------------------------------
def register_and_promote_model(logged_model_uri, model_name, validation_passed=True):
    client = MlflowClient()

    # Ensure registered model exists
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    # Register model
    try:
        rm = mlflow.register_model(logged_model_uri, model_name)
        version = rm.version
    except Exception:
        # Fallback for older MLflow versions
        rm2 = client.create_model_version(
            name=model_name,
            source=logged_model_uri,
            run_id=None
        )
        version = rm2.version

    print(f" Registered model {model_name} version {version}")

    # Small delay for MLflow backend
    time.sleep(2)

    # Move to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )
    print(f"Moved to Staging: {model_name} v{version}")

    # Promote to Production if validation passed
    if validation_passed:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        print(f" Moved to Production: {model_name} v{version}")
    else:
        print("Validation failed; kept in Staging")


# -------------------------------
# Training Function
# -------------------------------
def train(input_path, params_path):
    # Load params from YAML
    with open(params_path) as f:
        params = yaml.safe_load(f)

    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    spark_conf = params.get("spark", {})


    spark = (SparkSession.builder
             .appName("TitanicModelTraining")
             .config("spark.executor.cores", spark_conf.get("executor_cores", 2))
             .config("spark.executor.memory", spark_conf.get("executor_memory", "4g"))
             .config("spark.executor.instances", spark_conf.get("executor_instances", 2))
             .getOrCreate()
             )

    # Ensure path exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist.")
    df = spark.read.parquet(os.path.join(input_path, "data"))

    # Split dataset
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=params['train']['random_seed'])

    # Random Forest
    rf = RandomForestClassifier(
        featuresCol="features", 
        labelCol="label", 
        seed=params['train']['random_seed']
    )

    # Hyperparameter grid
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, params['train']['rf']['numTrees'])
                 .addGrid(rf.maxDepth, params['train']['rf']['maxDepth'])
                 .addGrid(rf.featureSubsetStrategy, params['train']['rf']['featureSubsetStrategy'])
                 .build())

    # Evaluators
    auc_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

    # Cross-validator
    crossval = CrossValidator(
        estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=auc_eval,
        numFolds=params['train']['crossval']['folds'],
        parallelism=params['train']['crossval']['parallelism'],
        seed=params['train']['random_seed']
    )

    with mlflow.start_run(log_system_metrics=True) as run:
        start_time = time.time()
        cvModel = crossval.fit(train_data)
        bestModel = cvModel.bestModel
        predictions = bestModel.transform(test_data)

        # Metrics
        auc = auc_eval.evaluate(predictions)
        accuracy = acc_eval.evaluate(predictions)
        preds_pd = predictions.select("label", "prediction").toPandas()
        precision = precision_score(preds_pd['label'], preds_pd['prediction'])
        recall = recall_score(preds_pd['label'], preds_pd['prediction'])
        f1 = f1_score(preds_pd['label'], preds_pd['prediction'])
        # Training time
        training_time = time.time() - start_time
        proba_pd = predictions.select("label", "probability").toPandas()
        logloss = log_loss(proba_pd['label'], proba_pd['probability'].apply(lambda x: x[1]))

        # Log parameters
        mlflow.log_param("Best Model maxDepth", bestModel.getOrDefault('maxDepth'))
        mlflow.log_param("Best Model numTrees", bestModel.getNumTrees)

        # Extra params (example)
        mlflow.log_param("train_split", "80/20")
        mlflow.log_param("random_seed", params['train']['random_seed'])
        # Log metrics
        mlflow.log_metric("val_auc", float(auc))
        mlflow.log_metric("val_accuracy", float(accuracy))
        mlflow.log_metric("val_precision", float(precision))
        mlflow.log_metric("val_recall", float(recall))
        mlflow.log_metric("val_f1_score", float(f1))
        mlflow.log_metric("system/cpu_percent", psutil.cpu_percent())
        mlflow.log_metric("system/memory_percent", psutil.virtual_memory().percent)
        mlflow.log_metric("training_time_sec", float(training_time))

        mlflow.log_metric("log_loss", float(logloss))

        # Extract true labels and positive-class probabilities
        y_true = proba_pd['label']
        y_score = proba_pd['probability'].apply(lambda x: float(x[1]))  # extract prob for class=1

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc_val = roc_auc_score(y_true, y_score)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.2f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("mlflow_img/roc_curve.png")

        # Log to MLflow
        mlflow.log_artifact("mlflow_img/roc_curve.png")
        mlflow.log_metric("roc_auc", roc_auc_val)

        # Confusion matrix
        cm = confusion_matrix(preds_pd['label'], preds_pd['prediction'])
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.savefig('mlflow_img/confusion_matrix.png')
        mlflow.log_artifact('mlflow_img/confusion_matrix.png')

        # Feature importances
        try:
    # Extract feature importances
            importances = bestModel.featureImportances.toArray()
            feature_cols = ["Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "EmbarkedIndex"]

            # Put into DataFrame
            fi_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": importances
            }).sort_values(by="importance", ascending=False)

            # Plot bar chart
            plt.figure(figsize=(8, 5))
            plt.barh(fi_df["feature"], fi_df["importance"], color="skyblue")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.title("Feature Importances")
            plt.gca().invert_yaxis()  # largest on top
            plt.tight_layout()

            # Save & log to MLflow
            plt.savefig("mlflow_img/feature_importances.png")
            mlflow.log_artifact("mlflow_img/feature_importances.png")
            plt.close()

        except Exception as e:
            print("Feature importance plotting failed:", e)

        # Log model
        mlflow.spark.log_model(bestModel, artifact_path="random_forest_model")

        # Register + promote model
        logged_model_uri = f"runs:/{run.info.run_id}/random_forest_model"
        register_and_promote_model(logged_model_uri, params['mlflow']['model_name'])


    spark.stop()


# -------------------------------
#  CLI Entry
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Path to preprocessed Parquet folder")
    parser.add_argument('--params', required=True, help="Path to params.yaml")
    args = parser.parse_args()
    train(args.input, args.params)
