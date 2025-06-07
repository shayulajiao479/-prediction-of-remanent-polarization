import pandas as pd
from sklearn.preprocessing import StandardScaler


data_real = pd.read_excel('67.xlsx')
data_synthetic = pd.read_excel('20000samples.xlsx')


value_columns = data_real.columns[1:]  # 假设所有列除了 'target' 是数值特征


scaler = StandardScaler()
data_real_scaled = data_real.copy()
data_real_scaled[value_columns] = scaler.fit_transform(data_real[value_columns])


data_synthetic_scaled = data_synthetic.copy()
data_synthetic_scaled[value_columns] = scaler.fit_transform(data_synthetic[value_columns])


sns_data_real = data_real_scaled.melt(id_vars=['target'], value_vars=value_columns,
                                      var_name='column', value_name='value')
sns_data_real['source'] = 'Real Data'

sns_data_synthetic = data_synthetic_scaled.melt(id_vars=['target'], value_vars=value_columns,
                                                var_name='column', value_name='value')
sns_data_synthetic['source'] = 'Synthetic Data'

sns_data = pd.concat([sns_data_real, sns_data_synthetic])

