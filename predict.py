"""
Inference script for predicting student grades using trained model
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from data_pipeline import DataPipeline
import json
import os

class StudentGradePredictor(nn.Module):
    """
    Feedforward Neural Network for predicting student grades
    (Same architecture as training script)
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(StudentGradePredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build the network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

class GradePredictor:
    """
    Class for loading trained model and making predictions
    """
    def __init__(self, model_path='best_student_grade_model.pt',
                 preprocessing_dir='preprocessing_artifacts'):
        self.model_path = model_path
        self.preprocessing_dir = preprocessing_dir
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = DataPipeline(save_dir=preprocessing_dir)

        self._load_model()

    def _load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_config = checkpoint['model_config']

        # Recreate model with same architecture
        self.model = StudentGradePredictor(
            input_dim=model_config['input_dim'],
            hidden_dims=model_config['hidden_dims'],
            dropout_rate=model_config['dropout_rate']
        )

        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully from {self.model_path}")
        print(f"Model architecture: {model_config}")

    def predict_from_file(self, file_path, save_results=True):
        """
        Predict grades for students in an Excel file

        Args:
            file_path: Path to Excel file with student data
            save_results: Whether to save predictions to file

        Returns:
            pandas.DataFrame: Original data with predictions
        """
        print(f"Loading data from {file_path}...")

        # Load and preprocess data
        X_processed = self.pipeline.prepare_inference_data(file_path)

        # Make predictions
        with torch.no_grad():
            X_tensor = X_processed.to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        # Load original data for results
        original_data = pd.read_excel(file_path)

        # Add predictions to original data
        results_df = original_data.copy()
        results_df['Predicted_Grade'] = predictions
        results_df['Predicted_Grade'] = results_df['Predicted_Grade'].round(2)

        # Calculate confidence intervals (rough approximation)
        # In practice, you'd want to implement proper uncertainty quantification
        results_df['Prediction_Confidence'] = 'High'  # Placeholder

        print(f"Generated predictions for {len(results_df)} students")
        print(f"Average predicted grade: {predictions.mean():.2f}")
        print(f"Grade range: {predictions.min():.2f} - {predictions.max():.2f}")

        if save_results:
            output_path = file_path.replace('.xlsx', '_predictions.xlsx')
            results_df.to_excel(output_path, index=False)
            print(f"Results saved to {output_path}")

        return results_df

    def predict_single_student(self, student_data):
        """
        Predict grade for a single student

        Args:
            student_data: Dictionary with student features

        Returns:
            float: Predicted grade
        """
        # Convert to DataFrame
        df = pd.DataFrame([student_data])

        # Save temporarily to use existing pipeline
        temp_file = 'temp_student.xlsx'
        df.to_excel(temp_file, index=False)

        try:
            # Use existing prediction pipeline
            X_processed = self.pipeline.prepare_inference_data(temp_file)

            with torch.no_grad():
                X_tensor = X_processed.to(self.device)
                prediction = self.model(X_tensor).cpu().numpy()[0]

            return round(float(prediction), 2)

        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def batch_predict(self, student_list):
        """
        Predict grades for multiple students

        Args:
            student_list: List of dictionaries with student features

        Returns:
            list: List of predicted grades
        """
        df = pd.DataFrame(student_list)
        temp_file = 'temp_students.xlsx'
        df.to_excel(temp_file, index=False)

        try:
            X_processed = self.pipeline.prepare_inference_data(temp_file)

            with torch.no_grad():
                X_tensor = X_processed.to(self.device)
                predictions = self.model(X_tensor).cpu().numpy()

            return [round(float(pred), 2) for pred in predictions]

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def get_feature_importance_simulation(self, baseline_student, feature_variations=None):
        """
        Simulate feature importance by varying individual features
        This is a simple approach - for production, use proper feature importance methods

        Args:
            baseline_student: Dictionary with baseline student features
            feature_variations: Dictionary specifying how to vary each feature

        Returns:
            Dictionary with feature impact analysis
        """
        if feature_variations is None:
            # Default variations for common features
            feature_variations = {
                'age': [15, 16, 17, 18, 19, 20],
                'studytime': [1, 2, 3, 4],
                'failures': [0, 1, 2, 3],
                'absences': [0, 5, 10, 15, 20, 25]
            }

        baseline_prediction = self.predict_single_student(baseline_student)
        feature_impacts = {}

        for feature, values in feature_variations.items():
            if feature in baseline_student:
                impacts = []
                for value in values:
                    modified_student = baseline_student.copy()
                    modified_student[feature] = value
                    prediction = self.predict_single_student(modified_student)
                    impact = prediction - baseline_prediction
                    impacts.append({'value': value, 'prediction': prediction, 'impact': impact})

                feature_impacts[feature] = {
                    'baseline_value': baseline_student[feature],
                    'baseline_prediction': baseline_prediction,
                    'variations': impacts
                }

        return feature_impacts

def demo_predictions():
    """Demonstrate the prediction functionality"""
    print("=== Student Grade Prediction Demo ===")

    # Initialize predictor
    try:
        predictor = GradePredictor()
    except FileNotFoundError:
        print("Error: Trained model not found. Please run the training script first.")
        return

    # Example 1: Single student prediction
    print("\n1. Single Student Prediction:")
    example_student = {
        'school': 'GP',
        'sex': 'F',
        'age': 17,
        'address': 'U',
        'famsize': 'GT3',
        'Pstatus': 'T',
        'Medu': 4,
        'Fedu': 4,
        'traveltime': 2,
        'studytime': 3,
        'failures': 0,
        'schoolsup': 'yes',
        'famsup': 'yes',
        'paid': 'no',
        'activities': 'yes',
        'nursery': 'yes',
        'higher': 'yes',
        'internet': 'yes',
        'romantic': 'no',
        'famrel': 4,
        'freetime': 3,
        'goout': 2,
        'Dalc': 1,
        'Walc': 1,
        'health': 5,
        'absences': 2,
        'G1': 15,
        'G2': 16,
        'Mjob': 'teacher',
        'Fjob': 'teacher',
        'reason': 'reputation',
        'guardian': 'mother'
    }

    predicted_grade = predictor.predict_single_student(example_student)
    print(f"Predicted grade for example student: {predicted_grade}")

    # Example 2: Feature impact analysis
    print("\n2. Feature Impact Analysis:")
    impacts = predictor.get_feature_importance_simulation(example_student)

    for feature, analysis in impacts.items():
        print(f"\n{feature.upper()}:")
        print(f"  Baseline: {analysis['baseline_value']} → {analysis['baseline_prediction']:.2f}")
        print("  Variations:")
        for var in analysis['variations']:
            print(f"    {var['value']} → {var['prediction']:.2f} (impact: {var['impact']:+.2f})")

    print("\n3. Batch Prediction Example:")
    # Example students with different characteristics
    students = [
        {**example_student, 'studytime': 1, 'failures': 2, 'absences': 10},
        {**example_student, 'studytime': 4, 'failures': 0, 'absences': 0},
        {**example_student, 'age': 19, 'studytime': 2, 'Dalc': 3, 'Walc': 3}
    ]

    predictions = predictor.batch_predict(students)
    print("Batch predictions:")
    for i, pred in enumerate(predictions):
        print(f"  Student {i+1}: {pred}")

if __name__ == "__main__":
    demo_predictions()
