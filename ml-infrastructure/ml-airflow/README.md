# ML Pipeline using Apache Airflow

### 1. Install Docker

### 2. Create necessary directories for airflow inside the project 
- /dags, /plugins, /logs etc as necessary

### 3. Download latest docker-compose.yaml and place it in the project
- Ref: https://airflow.apache.org/docs/apache-airflow/2.10.4/docker-compose.yaml

### 4. Install and setup Airflow

- Create a virtual environment and activate it
    ```
    python -m venv airflow_env
    source airflow_env/bin/activate
    ```

- Install Apache Airflow
    ```
    pip install apache-airflow
    ```

- Set Airflow home to this directory 
    ```
    export AIRFLOW_HOME=~/<current-proj-path>/airflow
    ```

- Start all services
  ```
  docker-compose build
  docker-compose up
  ```

- # To Stop airflow
  ```
  docker-compose down
  ```

### 5. Access the UI:
  ```
  Go to http://localhost:8080
  Login with:
  
  Username: <from docker-compose.yaml>
  Password: <from docker-compose.yaml>
  ```

