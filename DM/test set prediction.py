import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

data = pd.read_excel('64-6tz.xlsx')

X = data.iloc[:, 1:]  # 后几列是特征
y = data.iloc[:, 0].values.reshape(-1, 1)  # 第一列是target，并转为二维

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

model = MultiOutputRegressor(XGBRegressor(n_estimators=10, max_depth=8, random_state=42))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred, label='Dielectric constant', s=60)
plt.plot([0, 30], [0, 30], 'r-', linewidth=3, label='y = x')

# 计算R²（单目标时y_pred为二维，需要.ravel()或[:,0]）
r2 = r2_score(y_test, y_pred)
plt.text(0.1, 0.9, f'R² = {r2:.4f}', transform=plt.gca().transAxes)

plt.xlim(0, 30)
plt.ylim(0, 30)
plt.xlabel('Measured Value')
plt.ylabel('Predicted Value')

# 手动加粗所有边框和刻度线（增强效果）
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.tick_params(width=2)
plt.tight_layout()
plt.show()
