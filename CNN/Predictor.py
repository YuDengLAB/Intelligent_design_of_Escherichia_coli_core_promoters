import numpy as np
import tensorflow as tf
import os

def seq2onehot(seq):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return np.array([[mapping[base] for base in s] for s in seq])

model_path = 'weight_MSE_c.v-n.h5'
model = tf.keras.models.load_model(model_path)
print("The model is loaded.")

with open('./seq/csv/1280.txt', 'r') as file:
    dna_sequences = file.readlines()

one_hot_encoded_sequences = seq2onehot([seq.strip() for seq in dna_sequences])
one_hot_encoded_sequences = one_hot_encoded_sequences.reshape(-1, 29, 1, 4)  # reshape to fit the model input shape

predictions = model.predict(one_hot_encoded_sequences)

original_scale_predictions = np.exp(predictions) - 1

with open('./CNN_predictions_1280.txt', 'w') as file:
    for seq, pred in zip(dna_sequences, original_scale_predictions):
        file.write(f"Sequence: {seq.strip()} - Prediction: {pred[0]}\n")  # pred[0]假设预测结果是一维数组

print("The forecast results have been saved to a file.")
