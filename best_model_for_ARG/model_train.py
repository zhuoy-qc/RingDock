# final_tune_lean_features.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer
import xgboost as xgb
import joblib
import optuna

SEED = 42
np.random.seed(SEED)
os.makedirs('xgboost_tuning', exist_ok=True)

# ------------------ Load & Engineer Features ------------------
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
    return df

df = engineer_features(df)

# Smart feature set: RFECV rank=1 + rank=2 (delta_z_norm)
FEATURE_COLS = [
    'delta_z', 'distance', 'inv_distance', 'dihedral_angle',
    'distance_sq', 'delta_x_norm', 'sin_dihedral', 'cos_dihedral',
    'delta_z_norm'  # rank=2 — keep it!
]

X = df[FEATURE_COLS]
y = df['final_energy']

# ------------------ Stratified Split ------------------
def create_stratify_labels(X, n_bins=10):
    disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    strat = disc.fit_transform(X[['distance', 'dihedral_angle']])
    return (strat[:, 0] * n_bins + strat[:, 1]).astype(int)

strat_labels = create_stratify_labels(X)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, stratify=strat_labels, random_state=SEED
)

# ------------------ Optuna Objective (5-Fold CV) ------------------
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'random_state': SEED,
        'n_jobs': -1,
        # Conservative ranges based on your previous best
        'n_estimators': trial.suggest_int('n_estimators', 900, 1300),
        'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.08),  # lower!
        'max_depth': trial.suggest_int('max_depth', 3, 5),                # cap at 5
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 0.1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 0.1, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10)
    }
    model = xgb.XGBRegressor(**params)
    mse_scores = -cross_val_score(
        model, X_train_val, y_train_val,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    return mse_scores.mean()

# ------------------ Run Optimization ------------------
print("🔍 Tuning on lean feature set (9 features) with conservative ranges...")

study = optuna.create_study(direction='minimize')
# Warm-start with your original best (adjusted for 9 features)
warm_start = {
    'n_estimators': 1112,
    'learning_rate': 0.0597,
    'max_depth': 3,
    'subsample': 0.883,
    'colsample_bytree': 0.776,
    'reg_alpha': 0.0027,
    'reg_lambda': 0.00015,
    'min_child_weight': 1.0
}
study.enqueue_trial(warm_start)

study.optimize(objective, n_trials=50, n_jobs=-1)

print(f"\n✅ Best CV MSE: {study.best_value:.8f}")
print("Best params:", study.best_params)

# ------------------ Final Evaluation ------------------
best_params = study.best_params
best_params.update({
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': SEED,
    'n_jobs': -1
})

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train_val, y_train_val)

y_pred = final_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print(f"\n🎯 Final Test MSE: {mse_test:.8f}")
print(f"🎯 Final Test R²:  {r2_test:.6f}")

# ------------------ Save ------------------
save_path = 'xgboost_tuning/best_xgboost_final_lean.pkl'
joblib.dump({
    'model': final_model,
    'mse_test': mse_test,
    'r2_test': r2_test,
    'cv_mse': study.best_value,
    'feature_cols': FEATURE_COLS,
    'best_params': study.best_params
}, save_path)

print(f"\n📁 Model saved to: {save_path}")

# ------------------ Compare to Previous Best ------------------
PREV_BEST_MSE = 0.00575211
if mse_test < PREV_BEST_MSE:
    print(f"\n🎉 NEW BEST! Improved by {(PREV_BEST_MSE - mse_test)/PREV_BEST_MSE*100:.2f}%")
else:
    print(f"\nℹ️  Slightly worse than previous best ({PREV_BEST_MSE:.8f}). Consider using that model.")
