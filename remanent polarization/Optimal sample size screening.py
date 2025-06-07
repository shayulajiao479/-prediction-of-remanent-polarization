
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2

# 文件名和对应数据量
file_info = [
    ("/file.xlsx", 0),
    ("10000/file.xlsx", 10000),
    ("20000/file.xlsx", 20000),
    ("30000/file.xlsx", 30000),
    ("50000/file.xlsx", 50000),
]

rmse_list, mae_list, r2_list, sample_sizes = [], [], [], []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
reference_features = None  # 用于特征对齐

for idx, (filename, n_samples) in enumerate(file_info, 1):
    print(f"\n[{idx}/{len(file_info)}] Processing file: {filename} (Generated samples: {n_samples})")
    data = pd.read_excel(filename)

    # 第一次循环记录特征名，后续严格对齐
    if reference_features is None:
        feature_columns = [col for col in data.columns if col != "target"]
        reference_features = feature_columns
    else:
        feature_columns = [col for col in data.columns if col != "target"]
        if set(feature_columns) != set(reference_features):
            raise ValueError(f"Feature names in {filename} do not match the reference features!\n"
                             f"Expected: {reference_features}\nFound: {feature_columns}")
        # 顺序一致
        data = data[reference_features + ['target']]

    X = data[reference_features]
    y = data["target"]

    # 10折CV for RMSE, MAE, R2
    rmse_fold, mae_fold, r2_fold = [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse_fold.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae_fold.append(mean_absolute_error(y_val, y_pred))
        r2_fold.append(r2_score(y_val, y_pred))
        print(f"  Fold {fold}/10 finished. RMSE={rmse_fold[-1]:.4f}, MAE={mae_fold[-1]:.4f}, R2={r2_fold[-1]:.4f}")

    avg_rmse = np.mean(rmse_fold)
    avg_mae = np.mean(mae_fold)
    avg_r2 = np.mean(r2_fold)
    rmse_list.append(avg_rmse)
    mae_list.append(avg_mae)
    r2_list.append(avg_r2)

    print(f"Completed {filename}: Mean CV-RMSE={avg_rmse:.4f}, CV-MAE={avg_mae:.4f}, CV-R2={avg_r2:.4f}")
    sample_sizes.append(n_samples)
