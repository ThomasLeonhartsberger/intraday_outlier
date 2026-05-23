# Intraday Outlier Detection

This repository contains the code developed for a master's thesis on detecting
outliers in Austrian intraday electricity prices using machine learning models
and fundamental market data.

The main idea is to model the Austrian ID1 intraday price relative to the
day-ahead price and then use the model residuals to identify unusual price
situations. The project includes data preparation, model benchmarking, residual
analysis, and a small Streamlit prototype for exploring detected outliers.

## Repository Structure

```text
.
├── notebooks/          # Jupyter notebooks for data preparation, modelling, and analysis
├── src/                # Reusable Python helper functions and data preparation code
├── app.py              # Streamlit application for the anomaly explorer prototype
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Docker Compose setup for running the application
├── requirements.txt    # Python dependencies
└── README.md
```

## Data and Models

The folders `data/` and `data_final/` are not included in this repository,
because the target price data used in the thesis is confidential.

Trained model files (folder `models/`) are also not included. To fully run the application, the
required input data and trained model files must be placed in the expected local
folders. The notebooks are able to create those if the target price data is provided.

## Running the Application with Docker

If the required data and trained model files are available locally, the
Streamlit application can be started with Docker Compose:

```bash
docker compose up --build
```

The application should then be available in the browser at:

```text
http://localhost:8501
```

To stop the container, use:

```bash
docker compose down
```

## Notes

This repository is mainly intended to document the code and workflow used for
the thesis. Since the underlying price data and trained models are not included,
the repository is not fully reproducible without the missing local files.
