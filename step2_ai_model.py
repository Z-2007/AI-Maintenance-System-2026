import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

columns = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]
df = pd.read_csv('train_FD001.txt', sep='\s+', names=columns)

engine_1 = df[df['engine_id'] == 1].copy()
features = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11']
X = engine_1[features]

model = IsolationForest(contamination=0.05, random_state=42)
engine_1['anomaly'] = model.fit_predict(X)

plt.figure(figsize=(12, 6))
normal_data = engine_1[engine_1['anomaly'] == 1]
plt.scatter(normal_data['cycle'], normal_data['sensor_2'], color='blue', label='Normal', alpha=0.6)

anomaly_data = engine_1[engine_1['anomaly'] == -1]
plt.scatter(anomaly_data['cycle'], anomaly_data['sensor_2'], color='red', label='Anomaly', marker='x', s=100)

plt.title('AI Anomaly Detection (Isolation Forest) - Engine 1')
plt.xlabel('Cycle (Time)')
plt.ylabel('Sensor 2 Reading')
plt.legend()
plt.grid(True)

plt.savefig('ai_anomaly_detection.png', dpi=300)
plt.show()