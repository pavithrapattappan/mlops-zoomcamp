import os
import pickle
import pandas as pd
import mlflow
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yaml

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

@task
def load_data(url=URL):
    df = pd.read_parquet(url)
    print(f"Loaded records: {len(df):,}")  # Q3 print
    return df

@task
def prep_data(df):
    df = df.copy()
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df["PULocationID"] = df["PULocationID"].astype(str)
    df["DOLocationID"] = df["DOLocationID"].astype(str)
    print(f"After prep: {len(df):,}")  # Q4 print
    return df

@task
def train_and_log(df):
    features = ["PULocationID", "DOLocationID"]
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_df[features].to_dict("records"))
    y_train = train_df["duration"].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(f"Intercept: {lr.intercept_:.2f}")  # Q5 print

    with mlflow.start_run() as run:
        mlflow.set_tag("orchestrator", "Prefect")
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("features", "+".join(features))

        # log model and preprocessor
        mlflow.sklearn.log_model(lr, artifact_path="model")
        with open("dv.pkl", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("dv.pkl")
        os.remove("dv.pkl")

        # Print model_size_bytes from MLmodel (Q6 print if present)
        mlmodel_local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run.info.run_id}/model/MLmodel"
        )
        try:
            with open(mlmodel_local_path, "r") as f:
                meta = yaml.safe_load(f)
            size = meta.get("model_size_bytes")
            if size is not None:
                print(f"MLmodel model_size_bytes: {size}")
        except Exception as e:
            print(f"Couldn't read model_size_bytes: {e}")

        return run.info.run_id

@task
def register_model(run_id, name="nyc_taxi_duration_lr"):
    # Requires a tracking server with a DB backend for the Model Registry
    result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=name)
    print(f"Registered as: {result.name} v{result.version}")
    return result

@flow
def main(register: bool = False):
    # By default, log to a local ./mlruns folder
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    df = load_data()
    df = prep_data(df)
    run_id = train_and_log(df)
    if register:
        register_model(run_id)

if __name__ == "__main__":
    main(register=False)
