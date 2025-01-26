from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def prepare_data():
    print("Preparing data...!")


def train_test_split():
    print("Training data...!")


def training_xg_boost_regressor():
    print("Training with XGBoost regressor...!")


def predict_on_test_data():
    print("Predict on test data...!")


def predict_prob_on_test_data():
    print("Predict Prob on test data...!")


def get_metrics():
    print("Get Metrics...!")


with DAG('ml-pipeline',
         start_date=datetime(2021, 1, 1),
         schedule_interval='@daily',
         catchup=False
) as dag:

    task_prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data
    )

    task_train_test_split = PythonOperator(
        task_id='train_test_split',
        python_callable=train_test_split
    )

    task_training_xg_boost_regressor = PythonOperator(
        task_id='training_xg_boost_regressor',
        python_callable=training_xg_boost_regressor
    )

    task_predict_on_test_data = PythonOperator(
        task_id='predict_on_test_data',
        python_callable=predict_on_test_data
    )

    task_predict_prob_on_test_data = PythonOperator(
        task_id='predict_prob_on_test_data',
        python_callable=predict_prob_on_test_data
    )

    task_get_metrics = PythonOperator(
        task_id='get_metrics',
        python_callable=get_metrics
    )

    task_prepare_data >> task_train_test_split >> task_training_xg_boost_regressor >> \
    task_predict_on_test_data >> task_predict_prob_on_test_data >> task_get_metrics