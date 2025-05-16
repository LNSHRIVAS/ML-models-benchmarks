# Benchmarking Machine Learning Algorithms on CPU vs GPU

This repository presents a comprehensive benchmarking study comparing the performance of classical and deep learning machine learning models across CPU, GPU, and AWS cloud environments.

This project was implemented and tested entirely on Google Colab using GPU-enabled runtimes (Tesla T4) and local CPU runtimes for comparison.

To run this project, simply open any notebook from the `notebooks/` folder in Google Colab. Setup instructions are provided below.

---

## Abstract

This project explores the speed, efficiency, and memory usage of various machine learning algorithms when executed on:
- Local CPU (Intel Xeon Platinum 8255C)
- Local GPU (NVIDIA Tesla T4)
- AWS SageMaker (NVIDIA Tesla V100 on `p3.2xlarge` instances)

We implemented models using libraries such as Scikit-learn, TensorFlow, PyTorch-TabNet, and cuML to examine how hardware choices affect training time, inference performance, and overall computational efficiency.

---

## Models Evaluated

| Model               | Type             | Library          | Dataset                              |
|--------------------|------------------|------------------|---------------------------------------|
| Logistic Regression| Classical ML     | Scikit-learn     | Synthetic/UCI                         |
| KNN                | Classical ML     | Scikit-learn/cuML| UCI, NYC Taxi Trip                    |
| XGBoost            | Gradient Boosting| XGBoost (CPU/GPU)| Santander Transactions, UCI          |
| KMeans             | Clustering       | Scikit-learn/cuML| Synthetic Vehicle Maintenance         |
| MLP                | Deep Learning    | TensorFlow       | Car Prices Dataset                    |
| TabNet             | Deep Learning    | PyTorch-TabNet   | Car Prices Dataset                    |
| CNN                | Deep Learning    | TensorFlow       | Synthetic                             |
| LSTM / BiLSTM      | Deep Learning    | TensorFlow       | Verdict Text Dataset                  |
| RNN                | Deep Learning    | TensorFlow       | Verdict Text Dataset                  |

---

## Performance Metrics

For each model and hardware configuration, the following metrics were recorded:
- Training Time (in seconds)
- Inference Time per sample (in milliseconds)
- Model Size (in megabytes)
- Speedup Ratio (GPU vs CPU)
- Accuracy or RMSE depending on the task

---

## Datasets Used

- Car Price Prediction: https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge
- NYC Yellow Taxi Trips: https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data
- Santander Transaction Data: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- Synthetic Predictive Maintenance Dataset: https://github.com/iDharshan/ML-Based-Vehicle-Predictive-Maintenance-System-with-Real-Time-Visualization
- Custom Verdict Text Dataset for NLP-based models (RNN/LSTM)

---

## Technologies Used

- Python 3.10+
- Scikit-learn
- TensorFlow / Keras
- PyTorch TabNet
- RAPIDS cuML (for GPU-accelerated KNN and KMeans)
- XGBoost (CPU and GPU variants)
- Google Colab with Tesla T4
- AWS SageMaker with Tesla V100

---

## How to Set Up and Run the Code

### Running on Google Colab (Recommended)

1. Open any `.ipynb` notebook from the GitHub repository.
2. Click “Open in Colab”.
3. Go to Runtime > Change runtime type > Select GPU.
4. Execute all cells in the notebook.

Each notebook contains both CPU and GPU benchmark sections. Outputs include training time, inference speed, and performance metrics.

### Running Locally (Optional)

1. Clone the repository:

2. Install dependencies:
   - pip install -r requirements.txt

3. Open the desired notebook and follow the instructions inside.

---

## Key Results Summary

- XGBoost: Training time reduced from 1859s (CPU) to 100s (GPU)
- KMeans & KNN: Training was 4–30× faster on GPU; inference time remained high for KNN
- TabNet: Achieved 16× faster training on GPU with similar inference speed across hardware
- MLP: Slight training speed improvement on GPU; CPU provided faster inference
- LSTM/BiLSTM: Large speedup on GPU for training; inference slightly slower due to small batch sizes

---

## Conclusion

- GPU acceleration provides major benefits for training deep learning models and tree-based ensembles like XGBoost.
- Inference speed does not always improve with GPU unless batch size is large or model complexity justifies parallelization overhead.
- AWS GPU instances (p3.2xlarge) deliver powerful performance but may not be cost-effective for simpler models.
- For smaller or classical models, CPU execution is often sufficient and more efficient in practice.

This project serves as a practical guideline for choosing the right compute platform for different machine learning workloads.

---

## GitHub Repository Submission

GitHub Repository: https://github.com/<your-username>/<your-repo-name>

This repository contains all relevant `.ipynb` notebooks, performance logs, and summary results. The link is also included in the final project PDF submitted per course requirements.

---

## References

1. Çolhak et al., GPU-Accelerated Machine Learning in IoV, arXiv:2504.01905  
2. Gyawali, CPU vs GPU for Deep Learning, arXiv:2309.02521  
3. Pangre et al., Hardware Benchmarks for DL, IJETR 2020  
4. Kaggle Datasets, UCI Archive, AWS Documentation

---

This repository demonstrates real-world benchmarking across multiple ML architectures and hardware backends, supporting more informed and resource-aware deployment decisions.

