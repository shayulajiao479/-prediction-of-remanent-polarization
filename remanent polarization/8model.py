import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

# 读取数据
data = pd.read_excel("/file.xlsx")
features = ['t', 'raa', 'enamb', 'enbmb', 'rba', 'vb', 'rdcea', 'rdceb']
targets = ['target']

X = data[features]
y = data[targets]


scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)


XY_scaled = np.hstack([X_scaled, y_scaled])


iso = IsolationForest(contamination=0.08, random_state=42)  # 可调
outlier_pred = iso.fit_predict(XY_scaled)
mask = outlier_pred == 1  # 1为正常样本


X = X[mask]
y = y[mask]


X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=None)

model_params = {
    "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=8, max_depth=10, random_state=42)),
    "DecisionTree": MultiOutputRegressor(DecisionTreeRegressor(max_depth=8, random_state=42)),
    "SVR": MultiOutputRegressor(SVR(C=1.0, epsilon=0.1)),
    "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=10, max_depth=8, random_state=42)),
    "LinearRegression": MultiOutputRegressor(LinearRegression()),
    "XGBoost": MultiOutputRegressor(XGBRegressor(n_estimators=10, max_depth=8, random_state=42)),
    "AdaBoost": MultiOutputRegressor(AdaBoostRegressor(n_estimators=10, random_state=42)),
    "Ridge": MultiOutputRegressor(Ridge(alpha=1.0)),
}

train_test_cv_error = []

for model_name, model in model_params.items():
    print(f"训练模型: {model_name}")

    # 训练集交叉验证
    kf_train = KFold(n_splits=10, shuffle=True, random_state=42)
    train_cv_rmse = []
    for train_idx, val_idx in kf_train.split(X_train):
        X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv).reshape(-1, 1)
        y_pred_original = scaler_y.inverse_transform(y_pred)
        y_val_original = scaler_y.inverse_transform(y_val_cv)

        rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
        train_cv_rmse.append(rmse)
    train_cv_error = np.mean(train_cv_rmse)

    # 测试集交叉验证
    kf_test = KFold(n_splits=10, shuffle=True, random_state=42)
    test_cv_rmse = []
    for train_idx, val_idx in kf_test.split(X_test):
        X_test_cv, X_val_cv = X_test[train_idx], X_test[val_idx]
        y_test_cv, y_val_cv = y_test[train_idx], y_test[val_idx]

        model.fit(X_test_cv, y_test_cv)
        y_pred = model.predict(X_val_cv).reshape(-1, 1)
        y_pred_original = scaler_y.inverse_transform(y_pred)
        y_val_original = scaler_y.inverse_transform(y_val_cv)

        rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
        test_cv_rmse.append(rmse)
    test_cv_error = np.mean(test_cv_rmse)


    train_test_cv_error.append({
        "Model": model_name,
        "Train_CV_Error": train_cv_error,
        "Test_CV_Error": test_cv_error
    })

