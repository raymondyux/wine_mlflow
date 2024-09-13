import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

logging.basicConfig(level = logging.WARN)
logger = logging.getLogger(__name__)


# Get the arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type = float, required = False, default = 0.5)
parser.add_argument("--l1_ratio", type = float, required = False, default = 0.5)
args = parser.parse_args()

# Evaluation Function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the data

    csv_url = 'https://raw.githubusercontent.com/myuser114/wine-dataset/main/winequality.csv'
    try:
        data = pd.read_csv(csv_url, sep=';')
    except Exception as e:
        logger.exception(
            f"Unable to download the dataset, check the internet connect. Error: %s", e
        )

    y = data['quality']
    X = data.drop('quality', axis = 1)

    # Split the data into training and test sets with 0.75 and 0.25 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    alpha = args.alpha
    l1_ratio = args.l1_ratio
    exp = mlflow.set_experiment(experiment_name="experiment_1")

    with mlflow.start_run(experiment_id=exp.experiment_id):
        lr = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, random_state = 42)
        lr.fit(X_train, y_train)

        predictions = lr.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, predictions)

        print(f"Elastic model (alpha: {alpha:f}, l1_ratio: {l1_ratio:f})")
        print(f"RMSE: {rmse}")
        print(f"RMSE: {mae}")
        print(f"RMSE: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(lr, "mymodel")