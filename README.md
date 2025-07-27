# ML-Based Credit Risk Scoring System

A machine learning application that predicts loan default risk using various ML models and provides interactive visualizations through a Streamlit interface.

## Features

- Data preprocessing and feature engineering
- Multiple model training (Logistic Regression and Random Forest)
- Automatic model selection based on performance
- Risk categorization (Low, Medium, High)
- Feature importance visualization using SHAP
- Interactive web interface using Streamlit

## Project Structure

├── data/           # Directory for dataset storage
├── models/         # Directory for saved model files
├── notebooks/      # Jupyter notebooks for analysis
├── src/            # Source code
│   ├── app.py              # Streamlit web application
│   ├── data_processor.py   # Data preprocessing module
│   ├── model_trainer.py    # Model training and selection
│   └── visualizer.py       # Visualization utilities
└── tests/          # Test files

PlainText

Open Folder


## Requirements

- Python 3.x
- pandas
- scikit-learn
- streamlit
- shap
- matplotlib
- seaborn
- numpy
- joblib

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
python -m streamlit run src/app.py
```

2. Upload your CSV dataset through the web interface
3. Select the target column for prediction
4. Click "Train Model" to start the analysis
5. View the results including:
   - Risk distribution visualization
   - Feature importance plot
   - Model predictions

## Data Format

The application expects a CSV file with:
- Numerical and/or categorical features
- Binary target variable (0/1 for non-default/default)

## Model Details

The system trains and compares two models:
- Logistic Regression: For linear decision boundaries
- Random Forest: For capturing non-linear patterns

The best performing model is automatically selected for predictions.

## Visualization

- SHAP (SHapley Additive exPlanations) values for feature importance
- Distribution of risk categories
- Data preview functionality

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License
