# Benchmarking Machine Learning Algorithms on CPU vs GPU

This project benchmarks various machine learning models across CPU and GPU hardware to evaluate differences in training speed, inference time, and model performance across different model types and data domains.

## 🚀 Project Summary
We evaluated the performance of multiple ML algorithms including:
- **MLP (Multi-Layer Perceptron)** for tabular data
- **TabNet** (deep learning model for tabular data)
- **Bidirectional LSTM** (RNN variant) for sequential text data

Each model was tested on:
- **Local CPU**
- **Local GPU (Google Colab - Tesla T4)**
- **AWS SageMaker instances (CPU/GPU)**

## 📊 Models Benchmarked
| Model           | Task Type     | Framework        | Dataset         |
|----------------|---------------|------------------|-----------------|
| MLP            | Regression    | TensorFlow       | Car Prices      |
| TabNet         | Regression    | PyTorch-TabNet   | Car Prices      |
| BiLSTM (RNN)   | Text Seq.     | TensorFlow       | Verdict Corpus  |

## 🧠 Key Metrics Tracked
- **Training Time** (in seconds)
- **Inference Time** (ms/sample)
- **Model Performance**: RMSE for regression, Accuracy for classification
- **Model Size** (MB)

## 📁 Folder Structure
