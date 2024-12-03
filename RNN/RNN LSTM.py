import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.integrate import simps
import os

def dna_sequence_to_one_hot(dna_sequence):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    one_hot_sequence = np.array([mapping[base] for base in dna_sequence])
    return one_hot_sequence

def load_excel_data(file_path):
    df = pd.read_excel(file_path)
    one_hot_encoded_sequences = np.array([dna_sequence_to_one_hot(seq) for seq in df['Promoter_x']])
    df['strength'] = np.log(df['strength'] + 1)  
    labels = df[['strength']].values
    return one_hot_encoded_sequences, labels

one_hot_encoded_sequences, labels = load_excel_data('./source data.xlsx')  # 替换为正确的文件路径

X_train, X_test, y_train, y_test = train_test_split(one_hot_encoded_sequences, labels, test_size=0.33, random_state=42)

X_all = np.concatenate((X_train, X_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

all_dataset = tf.data.Dataset.from_tensor_slices((X_all, y_all)).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128)

tf.config.run_functions_eagerly(True)

model_path = './trained_model-n'

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("The model is loaded.")
else:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, input_shape=(29, 4), return_sequences=True),  
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.LSTM(50), 
        tf.keras.layers.Dropout(0.2),  
        tf.keras.layers.Dense(1) 
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mae'],
                  run_eagerly=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    epochs = 100
    history = model.fit(all_dataset, epochs=epochs, callbacks=[early_stopping])

    model.save(model_path)
    print("The model has been trained and saved.")

y_pred = model.predict(X_test)

indices = np.random.choice(len(y_test), size=10000, replace=False)
y_test_sampled = y_test[indices]
y_pred_sampled = y_pred[indices]

r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")

plt.figure(figsize=(12, 6))
plt.scatter(y_test_sampled, y_pred_sampled, alpha=0.6)
plt.plot([min(y_test_sampled), max(y_test_sampled)], [min(y_test_sampled), max(y_test_sampled)], color='red', linestyle='--')
plt.title('Predicted vs True Values (Sampled)')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, color='blue')
plt.savefig('./scatter_plot.pdf')  
plt.show()

plt.figure(figsize=(12, 6))
y_test_sorted = np.sort(y_test.flatten())
y_pred_sorted = np.sort(y_pred.flatten())
plt.plot(y_test_sorted, np.linspace(0, 1, len(y_test_sorted), endpoint=False), label='True CDF')
plt.plot(y_pred_sorted, np.linspace(0, 1, len(y_pred_sorted), endpoint=False), label='Predicted CDF')
plt.title('CDF of True Values vs Predicted Values')
plt.xlabel('Values')
plt.ylabel('CDF')
plt.legend()
plt.savefig('./cdf_plot.pdf') 
plt.show()

ks_stat, ks_p_value = ks_2samp(y_test.flatten(), y_pred.flatten())
print(f"KS Statistic: {ks_stat}, p-value: {ks_p_value:.6e}")

emd_value = wasserstein_distance(y_test.flatten(), y_pred.flatten())
print(f"EMD Value: {emd_value}")

bin_edges = np.linspace(min(y_test_sorted[0], y_pred_sorted[0]), max(y_test_sorted[-1], y_pred_sorted[-1]), 101)
counts_actual, _ = np.histogram(y_test_sorted, bins=bin_edges, density=True)
cdf_actual = np.cumsum(counts_actual) / np.sum(counts_actual)
counts_predicted, _ = np.histogram(y_pred_sorted, bins=bin_edges, density=True)
cdf_predicted = np.cumsum(counts_predicted) / np.sum(counts_predicted)
cdf_area_diff = simps(np.abs(cdf_actual - cdf_predicted), bin_edges[:-1])
print(f"Area Difference between CDFs: {cdf_area_diff}")

