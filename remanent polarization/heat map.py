import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


plt.rc('font', family='Times New Roman')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


data = pd.read_excel('48.xlsx')
target = data.iloc[:, 0]  # 第一列是目标变量（target）
features = data.iloc[:, 2:52]  # 第三列到第五十二列是特征（feature）

# 标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
corr = features_scaled_df.corr(method='pearson')

fig, ax = plt.subplots(figsize=(20, 20))  # 调整大小
heatmap = sns.heatmap(
    corr,
    annot=False,
    cbar=True,
    vmax=1,
    vmin=-1,
    linewidths=1,
    linecolor='white',
    xticklabels=True,
    yticklabels=True,
    square=True,
    cmap="RdBu_r",
    cbar_kws={"shrink": 0.8, "orientation": "vertical", "label": "Correlation Coefficient"}
)

heatmapcb = heatmap.figure.colorbar(heatmap.collections[0])
heatmapcb.ax.tick_params(labelsize=36)  # 设置颜色条刻度标签大小
heatmapcb.set_ticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])  # 设置颜色条的刻度位置
heatmapcb.set_ticklabels(['-1', '-0.8', '-0.6', '-0.4', '-0.2', '0', '0.2', '0.4', '0.6', '0.8', '1'])  # 设置颜色条的标签


ax.xaxis.tick_top()
plt.xticks(rotation=90, fontsize=28, weight='bold')
plt.yticks(rotation=0, fontsize=28, weight='bold')

# 保存
plt.savefig('Pearson.png', dpi=600)
plt.show()


