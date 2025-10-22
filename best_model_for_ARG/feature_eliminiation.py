# feature_selection_rfecv.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import RFECV
import xgboost as xgb
import joblib

SEED = 42
np.random.seed(SEED)
os.makedirs('xgboost_tuning', exist_ok=True)

# ------------------ Load & Engineer Features (with VdW terms) ------------------
df = pd.read_csv('all_energies.csv')

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
    # Add VdW / LJ terms
    df['inv_r6'] = df['inv_distance'] ** 6
    df['inv_r12'] = df['inv_distance'] ** 12
    return df

df = engineer_features(df)

# Full feature list (14 base + 2 VdW = 16 total)
FULL_FEATURE_COLS = [
    'delta_z', 'delta_x', 'distance', 'angle', 'dihedral_angle',
    'inv_distance', 'distance_sq',
    'cos_angle', 'sin_angle', 'cos_dihedral', 'sin_dihedral',
    'delta_z_norm', 'delta_x_norm',
    'inv_r6', 'inv_r12'
]

X_full = df[FULL_FEATURE_COLS]
y = df['final_energy']

# ------------------ Stratified Split ------------------
def create_stratify_labels(X, n_bins=10):
    disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    strat = disc.fit_transform(X[['distance', 'angle']])
    return (strat[:, 0] * n_bins + strat[:, 1]).astype(int)

strat_labels = create_stratify_labels(X_full)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_full, y, test_size=0.1, stratify=strat_labels, random_state=SEED
)

# ------------------ Fixed High-Performance XGBoost Params ------------------
# Use your previously found best params (or re-tune once if desired)
BEST_PARAMS = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': SEED,
    'n_jobs': -1,
    'n_estimators': 1112,
    'learning_rate': 0.05972485058923855,
    'max_depth': 3,
    'subsample': 0.882869300193969,
    'colsample_bytree': 0.7755346887476092,
    'reg_alpha': 0.002655843450849988,
    'reg_lambda': 0.00015290959455830383
}

# ------------------ RFECV: Recursive Feature Elimination with CV ------------------
print("🔍 Running RFECV (Recursive Feature Elimination with Cross-Validation)...")

# Use 5-fold CV, minimize MSE
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
estimator = xgb.XGBRegressor(**BEST_PARAMS)

# RFECV will automatically find the best number of features
selector = RFECV(
    estimator=estimator,
    step=1,                     # Remove 1 feature at a time
    cv=cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    min_features_to_select=5    # Don't go below 5 features
)

selector.fit(X_train_val, y_train_val)

# ------------------ Results ------------------
selected_features = X_train_val.columns[selector.support_].tolist()
n_selected = len(selected_features)
best_cv_mse = -selector.cv_results_['mean_test_score'][n_selected - 1]

print(f"\n✅ Optimal number of features: {n_selected}")
print(f"✅ Estimated CV MSE with selected features: {best_cv_mse:.8f}")
print(f"✅ Selected features: {selected_features}")

# ------------------ Final Model Training & Test Evaluation ------------------
X_train_selected = selector.transform(X_train_val)
X_test_selected = selector.transform(X_test)

final_model = xgb.XGBRegressor(**BEST_PARAMS)
final_model.fit(X_train_selected, y_train_val)

y_pred = final_model.predict(X_test_selected)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print(f"\n🎯 Final Test MSE: {mse_test:.8f}")
print(f"🎯 Final Test R²:  {r2_test:.6f}")

# ------------------ Save Results ------------------
output_path = 'xgboost_tuning/best_xgboost_rfecv.pkl'
joblib.dump({
    'model': final_model,
    'selector': selector,               # Full RFECV object (for transform)
    'selected_features': selected_features,
    'mse_test': mse_test,
    'r2_test': r2_test,
    'cv_mse_estimate': best_cv_mse,
    'full_feature_list': FULL_FEATURE_COLS
}, output_path)

print(f"\n📁 Model and selector saved to: {output_path}")

# Optional: Print feature ranking
ranking = pd.DataFrame({
    'feature': FULL_FEATURE_COLS,
    'rank': selector.ranking_  # 1 = selected, higher = eliminated earlier
}).sort_values('rank')
print("\n📊 Feature Ranking (1 = most important / selected):")
print(ranking.to_string(index=False))
