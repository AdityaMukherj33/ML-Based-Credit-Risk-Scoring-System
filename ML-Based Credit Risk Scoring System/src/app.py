import streamlit as st
import pandas as pd
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualizer import Visualizer

st.title('Credit Risk Scoring System 💳')

# File upload
upload_file = st.file_uploader("Upload your CSV file", type=['csv'])

if upload_file is not None:
    # Initialize components
    processor = DataProcessor()
    trainer = ModelTrainer()
    visualizer = Visualizer()
    
    # Load and process data
    df = processor.load_data(upload_file)
    st.write("Data Preview:", df.head())
    
    # Select target column
    target_column = st.selectbox("Select target column:", df.columns)
    
    if st.button("Train Model"):
        # Prepare data
        processed_df = processor.preprocess_data(df)
        X_train, X_test, y_train, y_test = processor.prepare_features(processed_df, target_column)
        
        # Train model
        with st.spinner('Training models...'):
            trainer.train_models(X_train, y_train)
            predictions, risk_categories = trainer.predict_risk(X_test)
        
        # Display results
        st.success('Model trained successfully!')
        
        # Plot visualizations
        st.subheader('Risk Distribution')
        visualizer.plot_risk_distribution(risk_categories)
        
        st.subheader('Feature Importance')
        visualizer.plot_feature_importance(
            trainer.best_model[1],
            X_test,
            processed_df.drop(target_column, axis=1).columns
        )