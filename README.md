# Promoter Strength Prediction and De Novo Design Platforms

This repository includes two platforms aimed at predicting transcriptional strength and designing novel promoter sequences. The first platform focuses on **promoter strength prediction** using advanced deep learning models, while the second platform generates **de novo promoter generation** with target strengths.

## Overview

### 1. Promoter Strength Prediction Platform

We developed three deep learning models to predict the transcriptional strength of promoter sequences based on a synthetic promoter library containing 112,955 sequences. The models are evaluated for their accuracy and correlation with experimental data:

- **Convolutional Neural Network (CNN)**: 0.87 correlation
- **Transformer**: 0.76 correlation
- **Long Short-Term Memory (LSTM)**: 0.81 correlation

These models help predict the transcriptional strength of new promoter sequences, enabling the design of more efficient synthetic promoters.

### 2. De Novo Promoter Generation Platform

This platform focuses on generating novel promoter sequences with target transcriptional strengths. The sequence generation is performed using two generative models:

Once the promoter sequences are generated, we use the **Transformer-based prediction model** to predict their transcriptional strength. After reverse filtering, sequences with the target strengths are obtained.

- **WGAN-GP Model**: The correlation between predicted and experimental strength values is **0.88**.
- **Conditional Diffusion Model**: The correlation between predicted and experimental strength values is **0.95**.

## Installation

### Requirements

- `torch>=1.9.0`
- `torchvision>=0.10.0`
- `scipy>=1.6.0`
- `matplotlib>=3.4.0`
- `pandas>=1.2.0`
- `tqdm>=4.60.0`
- `seaborn>=0.11.0`
- `einops>=0.3.0`
- `Pillow>=8.1.0`
- `IPython>=7.25.0`
- `torchmetrics>=0.5.0`
- `livelossplot>=0.5.3`
- `numpy>=1.19.0`
- `tensorflow==2.8.0`  
- `scikit-learn>=0.24.0`  
