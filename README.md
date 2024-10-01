# Wine Classification ML API

## Description

The goal of this project is to present a simple application using:

- **Kedro** to organize data science code
- **MLflow** for model logging and registry
- **FastAPI** to serve the trained model

The application was built using the Wine dataset. **XGBoost** was the chosen algorithm to tackle this multi-label classification problem, with **GridSearchCV** employed to fine-tune the model parameters.

## Installation

1. To execute the notebook with the training code, you need to install the required dependencies of the Kedro project. 

2. To run the FastAPI container, you need to install **Docker**. Follow the [official Docker documentation](https://docs.docker.com/get-docker/) for installation instructions.

## Running the Notebook and Pipeline

The model was trained using the code available in the `train_wine_model.ipynb` notebook.

Alternatively, you can run the model as a Kedro pipeline by executing the following command:

```bash
kedro run --pipeline=train_wine_model
```

## Model Versioning

The model versioning can be seen at the MLflow UI.

```bash
mlflow ui
```

This will launch the MLflow interface, where you can explore the existing experiment and check the first version of the model, which is already loaded in the container.


## Running the API

To run the API, first build the Docker image and then run it. The API source code can be found in `/src/fastapi`.

1. Build the Docker image:

    ```bash
    docker build -t wine-api ./src/fastapi
    ```

2. Run the Docker container:

    ```bash
    docker run -d -p 8000:8000 wine-api
    ```

Once the container is up, the API will be available at [http://localhost:8000/](http://localhost:8000/).

## API Endpoints and Documentation

The available API endpoints, including their request and response formats, can be explored in the Swagger UI. Once the API is running, navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to view the interactive documentation.
