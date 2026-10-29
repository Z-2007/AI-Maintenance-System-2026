import pandas as pd
import matplotlib.pyplot as plt

# 1. 定义列名
columns = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]

# 2. 读取数据
try:
    df = pd.read_csv('train_FD001.txt', sep='\s+', names=columns)
    print("=== 数据前 5 行 ===")
    print(df.head())
except Exception as e:
    print("读取失败，请检查文件是否在文件夹里！错误信息：", e)
    exit()

# 3. 取出第1台发动机的数据
engine_1 = df[df['engine_id'] == 1]

# 4. 画图
plt.figure(figsize=(10, 5))
plt.plot(engine_1['cycle'], engine_1['sensor_2'], label='Sensor 2', color='blue')
plt.title('Engine 1 - Sensor 2 over Time')
plt.xlabel('Cycle (Time)')
plt.ylabel('Sensor Reading')
plt.legend()
plt.grid(True)

# 保存图片并显示
plt.savefig('engine1_sensor2_trend.png', dpi=300)
print("\n图表已保存为 engine1_sensor2_trend.png")
plt.show()