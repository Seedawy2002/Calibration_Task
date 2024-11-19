# Calibration Task API - Data Team

### Project Overview

This repository contains the API for a calibration task focused on evaluating and serving multiple machine learning models through a Flask-based API. The task involves training and evaluating models, calibrating predictions, and deploying a pipeline for inference as an API endpoint.

### Key Features

- **Model Training and Calibration**: Training and calibrating machine learning models for improved prediction accuracy and consistency.
- **Pipeline Development**: Deploying a model pipeline as an API endpoint.
- **API Integration**: Enabling model evaluation and predictions via a Flask-based API.
- **Testing Automation**: Includes a Postman collection to facilitate API testing.

### Directory Structure
```
├── LICENSE                                  # License details
├── README.md                                # Project documentation
├── requirements.txt                         # App requirements
├── Testing_API.postman_collection.json      # Postman collection for API testing
├── app.py                                   # Flask application for serving the API
├── calibrated_rf_model.pkl                  # Pre-trained calibrated random forest model
├── calibration-notebook.ipynb               # Jupyter notebook for model training and calibration
```

### Requirements

To install and run the project, ensure the following dependencies are available:
- **Python 3.x**
- Libraries:
  - Flask
  - pandas
  - pickle
  - json
  - collections (OrderedDict)

Install all required dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Application

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Use the following command:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask App**:
   Launch the application with:
   ```bash
   python app.py
   ```
   The API will be hosted at `http://127.0.0.1:5000/`.

### API Usage

#### Endpoint: `/predict`
- **Method**: POST
- **Description**: Evaluates models and returns predictions for provided meta-features.

#### Request Body Example:
```json
{
    "0": {
        "num_clients": 12,
        "Sum of Instances in Clients": 15000,
        "Max. Of Instances in Clients": 3500,
        "Min. Of Instances in Clients": 200,
        "Stddev of Instances in Clients": 550,
        "Average Dataset Missing Values %": 4.5,
        "Min Dataset Missing Values %": 0.5,
        "Max Dataset Missing Values %": 8.0,
        "Stddev Dataset Missing Values %": 2.0,
        "Average Target Missing Values %": 2.8,
        "Min Target Missing Values %": 0.8,
        "Max Target Missing Values %": 5.0,
        "Stddev Target Missing Values %": 1.2,
        "No. Of Features": 48,
        "No. Of Numerical Features": 25,
        "No. Of Categorical Features": 23,
        "Sampling Rate": 0.2,
        "Average Skewness of Numerical Features": 0.6,
        "Minimum Skewness of Numerical Features": -0.3,
        "Maximum Skewness of Numerical Features": 1.4,
        "Stddev Skewness of Numerical Features": 0.3,
        "Average Kurtosis of Numerical Features": 2.8,
        "Minimum Kurtosis of Numerical Features": 1.5,
        "Maximum Kurtosis of Numerical Features": 3.2,
        "Stddev Kurtosis of Numerical Features": 0.5,
        "Avg No. of Symbols per Categorical Features": 5.3,
        "Min. No. Of Symbols per Categorical Features": 3,
        "Max. No. Of Symbols per Categorical Features": 7,
        "Stddev No. Of Symbols per Categorical Features": 1.2,
        "Avg No. Of Stationary Features": 12.3,
        "Min No. Of Stationary Features": 8,
        "Max No. Of Stationary Features": 18,
        "Stddev No. Of Stationary Features": 2.1,
        "Avg No. Of Stationary Features after 1st order": 11.5,
        "Min No. Of Stationary Features after 1st order": 7,
        "Max No. Of Stationary Features after 1st order": 17,
        "Stddev No. Of Stationary Features after 1st order": 1.9,
        "Avg No. Of Stationary Features after 2nd order": 10.8,
        "Min No. Of Stationary Features after 2nd order": 6,
        "Max No. Of Stationary Features after 2nd order": 15,
        "Stddev No. Of Stationary Features after 2nd order": 1.7,
        "Avg No. Of Significant Lags in Target": 5.5,
        "Min No. Of Significant Lags in Target": 2,
        "Max No. Of Significant Lags in Target": 10,
        "Stddev No. Of Significant Lags in Target": 1.5,
        "Avg No. Of Insignificant Lags in Target": 3.5,
        "Max No. Of Insignificant Lags in Target": 8,
        "Min No. Of Insignificant Lags in Target": 1,
        "Stddev No. Of Insignificant Lags in Target": 1.2,
        "Avg. No. Of Seasonality Components in Target": 3.2,
        "Max No. Of Seasonality Components in Target": 6,
        "Min No. Of Seasonality Components in Target": 2,
        "Stddev No. Of Seasonality Components in Target": 1.1,
        "Average Fractal Dimensionality Across Clients of Target": 2.4,
        "Maximum Period of Seasonality Components in Target Across Clients": 10,
        "Minimum Period of Seasonality Components in Target Across Clients": 3,
        "Entropy of Target Stationarity": 1.2
    },
    "1": {
        "num_clients": 8,
        "Sum of Instances in Clients": 13000,
        "Max. Of Instances in Clients": 3000,
        "Min. Of Instances in Clients": 100,
        "Stddev of Instances in Clients": 400,
        "Average Dataset Missing Values %": 6.0,
        "Min Dataset Missing Values %": 1.5,
        "Max Dataset Missing Values %": 12.0,
        "Stddev Dataset Missing Values %": 3.0,
        "Average Target Missing Values %": 3.2,
        "Min Target Missing Values %": 1.0,
        "Max Target Missing Values %": 7.0,
        "Stddev Target Missing Values %": 2.0,
        "No. Of Features": 52,
        "No. Of Numerical Features": 35,
        "No. Of Categorical Features": 17,
        "Sampling Rate": 0.15,
        "Average Skewness of Numerical Features": 0.4,
        "Minimum Skewness of Numerical Features": -0.1,
        "Maximum Skewness of Numerical Features": 1.3,
        "Stddev Skewness of Numerical Features": 0.5,
        "Average Kurtosis of Numerical Features": 3.0,
        "Minimum Kurtosis of Numerical Features": 1.8,
        "Maximum Kurtosis of Numerical Features": 3.5,
        "Stddev Kurtosis of Numerical Features": 0.6,
        "Avg No. of Symbols per Categorical Features": 4.5,
        "Min. No. Of Symbols per Categorical Features": 2,
        "Max. No. Of Symbols per Categorical Features": 6,
        "Stddev No. Of Symbols per Categorical Features": 1.1,
        "Avg No. Of Stationary Features": 10.5,
        "Min No. Of Stationary Features": 6,
        "Max No. Of Stationary Features": 14,
        "Stddev No. Of Stationary Features": 2.0,
        "Avg No. Of Stationary Features after 1st order": 9.8,
        "Min No. Of Stationary Features after 1st order": 5,
        "Max No. Of Stationary Features after 1st order": 13,
        "Stddev No. Of Stationary Features after 1st order": 1.8,
        "Avg No. Of Stationary Features after 2nd order": 9.2,
        "Min No. Of Stationary Features after 2nd order": 4,
        "Max No. Of Stationary Features after 2nd order": 12,
        "Stddev No. Of Stationary Features after 2nd order": 1.6,
        "Avg No. Of Significant Lags in Target": 4.8,
        "Min No. Of Significant Lags in Target": 2,
        "Max No. Of Significant Lags in Target": 8,
        "Stddev No. Of Significant Lags in Target": 1.4,
        "Avg No. Of Insignificant Lags in Target": 3.2,
        "Max No. Of Insignificant Lags in Target": 6,
        "Min No. Of Insignificant Lags in Target": 1,
        "Stddev No. Of Insignificant Lags in Target": 1.0,
        "Avg. No. Of Seasonality Components in Target": 3.0,
        "Max No. Of Seasonality Components in Target": 5,
        "Min No. Of Seasonality Components in Target": 1,
        "Stddev No. Of Seasonality Components in Target": 1.0,
        "Average Fractal Dimensionality Across Clients of Target": 2.2,
        "Maximum Period of Seasonality Components in Target Across Clients": 9,
        "Minimum Period of Seasonality Components in Target Across Clients": 2,
        "Entropy of Target Stationarity": 1.1
    }
}
```

#### Response Body Example:
```json
{
    "0": {
        "LASSO": 0.271647,
        "ELASTICNETCV": 0.249047,
        "LinearSVR": 0.235947,
        "HUBERREGRESSOR": 0.145717,
        "QUANTILEREGRESSOR": 0.050852,
        "XGBRegressor": 0.04679
    },
    "1": {
        "LASSO": 0.288635,
        "ELASTICNETCV": 0.249782,
        "HUBERREGRESSOR": 0.233121,
        "LinearSVR": 0.136264,
        "QUANTILEREGRESSOR": 0.053012,
        "XGBRegressor": 0.039187
    }
}
```

### Notebook Details

The `calibration-notebook.ipynb` includes:
- Data preprocessing
- Model training and calibration steps
- Visualizations and performance evaluation metrics

### Testing

Use the provided `Testing_API.postman_collection.json` file to test the API in Postman. Import the collection into Postman and run the predefined tests.

### Important Notes

- Ensure the API is running before initiating requests.
- The model is calibrated for performance and interpretability.
- Results are dependent on the meta-features provided in the request.

### License

This project is licensed under the Apache-02 License. Refer to the [LICENSE](LICENSE) file for more details.

### Contact



---

This updated README includes the libraries you mentioned (`pickle`, `pandas`, `json`, `OrderedDict`) and aligns with your project requirements. Let me know if you need further refinements!
