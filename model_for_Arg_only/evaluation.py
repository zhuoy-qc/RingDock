import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Define file paths
model_path = 'model_output/final_model_20250917_194857.pkl'
data_path = 'cleaned.csv'
test_indices_path = 'model_output/test_indices.csv'

# Load the trained model
model = joblib.load(model_path)

# Load the dataset
df = pd.read_csv(data_path)
X = df[['delta_z', 'delta_x', 'dihedral_angle', 'distance']]
y = df['final_energy']

# Load test indices
test_indices_df = pd.read_csv(test_indices_path)
test_indices = test_indices_df['index'].values

# Prepare the test set
X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]

# Make predictions
y_pred = model.predict(X_test)

# Compute evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)  # Calculate Mean Absolute Error

# Compute absolute error for each sample
absolute_errors = np.abs(y_test.values - y_pred)

# Create a DataFrame with actual, predicted, and error values
results_df = pd.DataFrame({
    'Actual_final_energy': y_test.values,
    'Predicted_final_energy': y_pred,
    'Absolute_Error': absolute_errors
})

# Sort by absolute error in descending order and get top 10 samples
top_10_errors = results_df.nlargest(10, 'Absolute_Error')

# Print evaluation results
print("\n=== Test Set Evaluation ===")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Maximum Absolute Error: {absolute_errors.max():.6f}")

# Print top 10 samples with largest absolute errors
print("\nTop 10 Largest Absolute Errors:")
print(top_10_errors[['Actual_final_energy', 'Predicted_final_energy', 'Absolute_Error']])

# Print first 10 sample predictions (in original order)
print("\nFirst 10 Sample Predictions:")
print(results_df[['Actual_final_energy', 'Predicted_final_energy', 'Absolute_Error']].head(10))

# Print input features for top 10 error samples
print("\nInput Features for Top 10 Error Samples:")
print(X_test.iloc[top_10_errors.index][['delta_z', 'delta_x', 'dihedral_angle', 'distance']])

