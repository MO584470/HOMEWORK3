import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
file_path = r"D:\LSTM-Multivariate_pollution.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("数据集基本信息")
print("=" * 60)
print(f"数据形状: {df.shape}")
print(f"\n列名: {df.columns.tolist()}")
print(f"\n前5行数据:")
print(df.head())

# 2. 数据预处理
# 转换日期列
df['date'] = pd.to_datetime(df['date'])

# 目标列是 'pollution'（PM2.5浓度）
target_col = 'pollution'

# 检查缺失值
print(f"\n缺失值统计:")
print(df.isnull().sum())

# 处理风向列（wnd_dir 是类别变量，做 one-hot 编码）
df = pd.get_dummies(df, columns=['wnd_dir'], prefix='wind')
print(f"\nOne-hot编码后的列名:")
print(df.columns.tolist())

# 3. 可视化探索
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

# PM2.5 时间序列（前1000小时）
axes[0, 0].plot(df['date'][:1000], df['pollution'][:1000], linewidth=0.8, color='blue')
axes[0, 0].set_title('PM2.5 浓度时间序列（前1000小时）')
axes[0, 0].set_xlabel('时间')
axes[0, 0].set_ylabel('PM2.5 (µg/m³)')
axes[0, 0].tick_params(axis='x', rotation=45)
# PM2.5 分布直方图
axes[0, 1].hist(df['pollution'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_title('PM2.5 浓度分布')
axes[0, 1].set_xlabel('PM2.5 (µg/m³)')
axes[0, 1].set_ylabel('频次')
# 温度 vs PM2.5
axes[0, 2].scatter(df['temp'][:5000], df['pollution'][:5000], alpha=0.3, s=1, color='red')
axes[0, 2].set_title('温度 vs PM2.5')
axes[0, 2].set_xlabel('温度 (°C)')
axes[0, 2].set_ylabel('PM2.5 (µg/m³)')
# 气压 vs PM2.5
axes[1, 0].scatter(df['press'][:5000], df['pollution'][:5000], alpha=0.3, s=1, color='purple')
axes[1, 0].set_title('气压 vs PM2.5')
axes[1, 0].set_xlabel('气压 (hPa)')
axes[1, 0].set_ylabel('PM2.5 (µg/m³)')
# 露点 vs PM2.5
axes[1, 1].scatter(df['dew'][:5000], df['pollution'][:5000], alpha=0.3, s=1, color='orange')
axes[1, 1].set_title('露点温度 vs PM2.5')
axes[1, 1].set_xlabel('露点温度 (°C)')
axes[1, 1].set_ylabel('PM2.5 (µg/m³)')
# 风速 vs PM2.5
axes[1, 2].scatter(df['wnd_spd'][:5000], df['pollution'][:5000], alpha=0.3, s=1, color='brown')
axes[1, 2].set_title('风速 vs PM2.5')
axes[1, 2].set_xlabel('风速 (m/s)')
axes[1, 2].set_ylabel('PM2.5 (µg/m³)')

plt.tight_layout()
plt.show()

# 4. 构造 LSTM 输入
def create_sequences(data, target_col, n_hours):
    X, y = [], []
    for i in range(len(data) - n_hours):
        X.append(data.iloc[i:i + n_hours].values)
        y.append(data.iloc[i + n_hours][target_col])
    return np.array(X), np.array(y)

# 选择特征列
exclude_cols = ['date']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n" + "=" * 60)
print(f"使用特征列 (共 {len(feature_cols)} 个):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

# 提取特征数据
data_features = df[feature_cols].copy()

# 归一化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_features)
scaled_df = pd.DataFrame(scaled_data, columns=feature_cols)

# 设置时间步长（本次实验用过去24小时预测下一小时）
n_hours = 24

# 构造序列
X, y = create_sequences(scaled_df, target_col=target_col, n_hours=n_hours)

print(f"\n序列构造完成:")
print(f"X 形状: {X.shape}  # (样本数={X.shape[0]}, 时间步长={n_hours}, 特征数={X.shape[2]})")
print(f"y 形状: {y.shape}  # (预测目标数={y.shape[0]})")

# 5. 按时间顺序划分数据集
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]

X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print(f"\n数据集划分:")
print(f"训练集: {X_train.shape[0]} 样本 ({train_size / len(X) * 100:.1f}%)")
print(f"验证集: {X_val.shape[0]} 样本 ({val_size / len(X) * 100:.1f}%)")
print(f"测试集: {X_test.shape[0]} 样本 ({(len(X) - train_size - val_size) / len(X) * 100:.1f}%)")

# 6. 构建 LSTM 模型
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(n_hours, X.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 7. 训练模型
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# 8. 评估模型

# 训练曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='训练损失', linewidth=1.5)
axes[0].plot(history.history['val_loss'], label='验证损失', linewidth=1.5)
axes[0].set_xlabel('训练轮次')
axes[0].set_ylabel('损失 (MSE)')
axes[0].set_title('模型损失曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='训练MAE', linewidth=1.5)
axes[1].plot(history.history['val_mae'], label='验证MAE', linewidth=1.5)
axes[1].set_xlabel('训练轮次')
axes[1].set_ylabel('平均绝对误差 (MAE)')
axes[1].set_title('MAE曲线')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 测试集预测
print("\n正在进行测试集预测...")
y_pred = model.predict(X_test)

# 逆变换得到真实 PM2.5 值
# 找到 'pollution' 在特征列中的索引
pm_idx = feature_cols.index(target_col)

def inverse_transform_pm(y_scaled, scaler, pm_idx):
    """将归一化的 PM2.5 还原为原始值"""
    dummy = np.zeros((len(y_scaled), len(feature_cols)))
    dummy[:, pm_idx] = y_scaled.flatten()
    dummy_inv = scaler.inverse_transform(dummy)
    return dummy_inv[:, pm_idx]

y_test_orig = inverse_transform_pm(y_test, scaler, pm_idx)
y_pred_orig = inverse_transform_pm(y_pred, scaler, pm_idx)

# 计算误差指标
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
r2 = r2_score(y_test_orig, y_pred_orig)
mape = np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 0.01))) * 100

print("\n" + "=" * 60)
print("测试集评估结果")
print("=" * 60)
print(f"MAE  (平均绝对误差):   {mae:.2f} µg/m³")
print(f"RMSE (均方根误差):    {rmse:.2f} µg/m³")
print(f"R²   (决定系数):      {r2:.4f}")
print(f"MAPE (平均百分比误差): {mape:.2f}%")

# 可视化预测结果
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 前500个测试样本
n_show = min(500, len(y_test_orig))
axes[0].plot(y_test_orig[:n_show], label='实际 PM2.5', alpha=0.7, linewidth=1, color='blue')
axes[0].plot(y_pred_orig[:n_show], label='预测 PM2.5', alpha=0.7, linewidth=1, color='red')
axes[0].set_xlabel('时间步（小时）')
axes[0].set_ylabel('PM2.5 (µg/m³)')
axes[0].set_title(f'LSTM 预测效果对比（前{n_show}小时）')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 散点图（实际 vs 预测）
axes[1].scatter(y_test_orig, y_pred_orig, alpha=0.3, s=5, color='green')
axes[1].plot([y_test_orig.min(), y_test_orig.max()],
             [y_test_orig.min(), y_test_orig.max()],
             'r--', linewidth=2, label='理想预测线')
axes[1].set_xlabel('实际 PM2.5 (µg/m³)')
axes[1].set_ylabel('预测 PM2.5 (µg/m³)')
axes[1].set_title(f'实际值 vs 预测值散点图 (R² = {r2:.4f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. 保存模型
model.save('lstm_pollution_predictor.h5')






