import tensorflow as tf
import numpy as np

trained_model = tf.keras.models.load_model('./trained_model-n.h5')

with open('./seq/csv/1280.txt', 'r') as file:
    dna_sequences = file.readlines()

def dna_sequence_to_one_hot(dna_sequence):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    one_hot_sequence = np.array([mapping[base] for base in dna_sequence])
    return one_hot_sequence

one_hot_encoded_sequences = np.array([dna_sequence_to_one_hot(seq.strip()) for seq in dna_sequences])

predictions = trained_model.predict(one_hot_encoded_sequences)

original_scale_predictions = np.exp(predictions) - 1

with open('./RNN_predictions_1280.txt', 'w') as file:
    for seq, pred in zip(dna_sequences, original_scale_predictions):
        file.write(f"Sequence: {seq.strip()} - Prediction: {pred[0]}\n")  

print("The forecast results have been saved to a file.")

