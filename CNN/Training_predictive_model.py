import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import ks_2samp, wasserstein_distance, pearsonr
from scipy.integrate import simps
from sklearn.metrics import r2_score
import os

# Define DNA sequence to one-hot encoding conversion
def seq2onehot(seq):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return np.array([[mapping[base] for base in s] for s in seq])

# Load data
x_train = np.load("./seq/npy/c.v_p.npy", allow_pickle=True)
x_train = seq2onehot(x_train)
x_train = x_train.reshape(-1, 29, 1, 4)  # reshape to fit the model input shape

# Load and convert y_train to float
y_train = np.load("./seq/npy/c.v_s.npy", allow_pickle=True)
y_train = y_train.astype(float)
y_train = np.log2(y_train + 1)

# Split data into training and testing sets (2/3 training, 1/3 testing)
train_size = int(2 * len(x_train) / 3)
train_feature = x_train[:train_size]
test_feature = x_train[train_size:]
train_label = y_train[:train_size]
test_label = y_train[train_size:]

# Define model save path
model_path = 'weight_MSE_c.v-n.h5'

# Check if model exists
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("The model is loaded.")
else:
    # Build the model
    model = Sequential()
    model.add(Conv2D(100, (6, 1), padding='same', input_shape=(29, 1, 4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(200, (5, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(200, (5, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the model using all data
    history = model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.1, shuffle=True, callbacks=[early_stopping])

    # Save model
    model.save(model_path)
    print("The model has been trained and saved.")

    # Plot training loss and MAE
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation Loss/MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / MAE')
    plt.legend()
    plt.savefig('./training_loss_mae.pdf')
    plt.show()

# Predict
y_pred = model.predict(test_feature, verbose=1)
y_pred = y_pred[:, 0]

# Calculate Pearson correlation coefficient
cor_pearsonr = pearsonr(y_pred, test_label)
print(cor_pearsonr)

# Randomly select 10000 points for scatter plot
indices = np.random.choice(len(test_label), size=10000, replace=False)
y_test_sampled = test_label[indices]
y_pred_sampled = y_pred[indices]

# Calculate R2
r2 = r2_score(test_label, y_pred)
print(f"R2 Score: {r2}")

# Plot scatter plot of predicted vs true values
plt.figure(figsize=(6.7, 6.7))
plt.scatter(y_test_sampled, y_pred_sampled, alpha=0.6)
plt.plot([min(y_test_sampled), max(y_test_sampled)], [min(y_test_sampled), max(y_test_sampled)], color='red', linestyle='--')
plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, color='black')
plt.title('Predicted vs True Values', fontsize  = 20)
plt.xlabel('True Values', fontsize  = 20)
plt.ylabel('Predicted Values', fontsize  = 20)
plt.tick_params(axis='both', which = 'major', labelsize = 20)
plt.savefig('./scatter_plot.pdf')
plt.show()

# Plot CDF
plt.figure(figsize=(16.75, 6.7))
y_test_sorted = np.sort(test_label)
y_pred_sorted = np.sort(y_pred)
plt.plot(y_test_sorted, np.linspace(0, 1, len(y_test_sorted), endpoint=False), label='True CDF')
plt.plot(y_pred_sorted, np.linspace(0, 1, len(y_pred_sorted), endpoint=False), label='Predicted CDF')
plt.title('CDF of True Values vs Predicted Values', fontsize  = 20)
plt.xlabel('Values', fontsize  = 20)
plt.ylabel('CDF', fontsize  = 20)
plt.tick_params(axis='both', which = 'major', labelsize = 20)
plt.legend()
plt.savefig('cdf_plot.pdf')
plt.show()

# KS test
ks_stat, ks_p_value = ks_2samp(test_label, y_pred)
print(f"KS Statistic: {ks_stat}, p-value: {ks_p_value:.6e}")

# Calculate EMD value
emd_value = wasserstein_distance(test_label, y_pred)
print(f"EMD Value: {emd_value}")

# Calculate area difference between aligned CDFs
common_bin_edges = np.linspace(min(np.min(test_label), np.min(y_pred)), max(np.max(test_label), np.max(y_pred)), 101)
counts_actual, _ = np.histogram(test_label, bins=common_bin_edges, density=True)
cdf_actual = np.cumsum(counts_actual) / np.sum(counts_actual)
counts_predicted, _ = np.histogram(y_pred, bins=common_bin_edges, density=True)
cdf_predicted = np.cumsum(counts_predicted) / np.sum(counts_predicted)
cdf_area_diff = simps(np.abs(cdf_actual - cdf_predicted), common_bin_edges[:-1])
print(f"Area Difference between CDFs: {cdf_area_diff}")

