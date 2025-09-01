
# First setup conda environment with python version 3.9
conda create --name $name_of_env python=3.9

# Install jdk version 17 as we are using 
conda install -c conda-forge openjdk=17

# Set JAVA_HOME to that path
echo $CONDA_PREFIX
export JAVA_HOME=$CONDA_PREFIX
export PATH=$JAVA_HOME/bin:$PATH

# to download which are required
pip install -r requirements.txt

# initialise git
git init
# initialise dvc
dvc init



# remote storage
mkdir -p /dvc_storage

# add dvc remote 
dvc remote add -d localremote /dvc_storage

# Commit config change to Git
git commit .dvc/config -m "Configure local DVC remote

# add train.csv to dvc
dvc add data/raw/train.csv

# To track the changes with git, run:
git add data/raw/.gitignore data/raw/train.csv.dvc

git commit -m "Track dataset mydata.csv with DVC"

dvc push

# creating the mlflow db
mkdir -p mlruns_db/artifacts
touch mlruns_db/mlflow.db

# run the server

nohup mlflow server --backend-store-uri sqlite:///./mlruns_db/mlflow.db --default-artifact-root ./mlruns_db/artifacts --host 0.0.0.0 --port 5000 &> mlflow.log &

# for initial training
dvc repro

# if want to only train
 dvc repro train

# if we want only preprocessing
dvc repro preprocess

# for drift detection
# for changing permissions
chmod +x src/train_pipeline.sh 

# for running drift detection
src/train_pipeline.sh new_file_path

# for fast api
python app.py

# after running the fast api run for calling the api
$ curl -X POST "http://127.0.0.1:8000/predict"   -H "Content-Type: application/json"   -d '{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
  }'

