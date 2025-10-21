# evaluate_best_model.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
import joblib

SEED = 42

# ------------------ Load and Engineer Features (same as training) ------------------
def engineer_features(df):
    df = df.copy()
    df['distance'] = df['distance'].clip(lower=1e-3)
    df['inv_distance'] = 1.0 / df['distance']
    df['distance_sq'] = df['distance'] ** 2
    df['angle_rad'] = np.radians(df['angle'])
    df['dihedral_rad'] = np.radians(df['dihedral_angle'])
    df['cos_angle'] = np.cos(df['angle_rad'])
    df['sin_angle'] = np.sin(df['angle_rad'])
    df['cos_dihedral'] = np.cos(df['dihedral_rad'])
    df['sin_dihedral'] = np.sin(df['dihedral_rad'])
    df['delta_z_norm'] = df['delta_z'] / df['distance']
    df['delta_x_norm'] = df['delta_x'] / df['distance']
    return df

# Load data
df = pd.read_csv('all_energies.csv')
df = engineer_features(df)

# ------------------ Recreate the EXACT same test set ------------------
# Use the same stratification and split logic as in training
def create_stratify_labels(X, n_bins=10):
    disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    strat = disc.fit_transform(X[['distance', 'dihedral_angle']])
    return (strat[:, 0] * n_bins + strat[:, 1]).astype(int)

# Feature columns used in final model
FEATURE_COLS = [
    'delta_z', 'distance', 'inv_distance', 'dihedral_angle',
    'distance_sq', 'delta_x_norm', 'sin_dihedral', 'cos_dihedral',
    'delta_z_norm'
]

X = df[FEATURE_COLS]
y = df['final_energy']

strat_labels = create_stratify_labels(X)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.1, stratify=strat_labels, random_state=SEED
)

# Keep original indices to trace back to full row
test_indices = X_test.index

# ------------------ Load Model ------------------
model_path = 'xgboost_tuning/best_xgboost_final_lean.pkl'
model_data = joblib.load(model_path)
model = model_data['model']

print(f"✅ Loaded model from: {model_path}")
print(f"✅ Model trained on features: {model_data['feature_cols']}")

# ------------------ Predict ------------------
y_pred = model.predict(X_test)

# ------------------ Compute Metrics ------------------
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("📊 MODEL PERFORMANCE ON TEST SET")
print("="*50)
print(f"Mean Squared Error (MSE): {mse:.8f}")
print(f"Mean Absolute Error (MAE): {mae:.8f}")
print(f"Coefficient of Determination (R²): {r2:.6f}")
print("="*50)

# ------------------ Find Largest Error ------------------
abs_errors = np.abs(y_test - y_pred)
max_error_idx_in_test = abs_errors.idxmax()  # index in y_test
max_error = abs_errors[max_error_idx_in_test]
true_val = y_test[max_error_idx_in_test]
pred_val = y_pred[y_test.index.get_loc(max_error_idx_in_test)]

# Get original row from full dataframe
original_row = df.loc[max_error_idx_in_test]

print(f"\n🚨 SAMPLE WITH LARGEST ABSOLUTE ERROR:")
print(f"   Absolute Error: {max_error:.6f}")
print(f"   True Energy:    {true_val:.6f}")
print(f"   Predicted:      {pred_val:.6f}")
print(f"\n   Full feature values for this sample:")
for col in FEATURE_COLS:
    print(f"     {col:20}: {original_row[col]:.6f}")
print(f"     final_energy    : {original_row['final_energy']:.6f}")

# Optional: Save this sample to a file
# outlier_row = df.loc[[max_error_idx_in_test]]
# outlier_row.to_csv('xgboost_tuning/largest_error_sample.csv', index=False)
# print(f"\n💾 Largest error sample saved to: xgboost_tuning/largest_error_sample.csv")
