# Benchmarking Machine Learning Algorithms on CPU vs GPU

This repository contains a comprehensive benchmarking study comparing the performance of classical and deep learning machine learning models across CPU, GPU, and AWS cloud environments.

## 📌 Abstract

This project explores the speed, efficiency, and memory usage of various ML algorithms when executed on:
- Local **CPU** (Intel Xeon Platinum 8255C)
- Local **GPU** (NVIDIA Tesla T4)
- **AWS SageMaker** (NVIDIA Tesla V100 on `p3.2xlarge` instances)

We implemented models using libraries like Scikit-learn, TensorFlow, PyTorch-TabNet, and cuML to examine how hardware choices affect ML workloads in training and inference phases.

---

## 🧠 Models Evaluated

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

## 📊 Performance Metrics

For each model and hardware configuration, we recorded:
-  **Training Time**
-  **Inference Time**
-  **Memory Usage**
-  **Speedup Ratios (GPU vs CPU)**

---

## 💾 Datasets Used

- [Car Price Prediction](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge)
- [NYC Yellow Taxi Trips](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data)
- [Santander Transaction Data](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Synthetic Predictive Maintenance Dataset](https://github.com/iDharshan/ML-Based-Vehicle-Predictive-Maintenance-System-with-Real-Time-Visualization)

---

## ⚙️ Technologies

- Python, NumPy, Pandas
- Scikit-learn
- TensorFlow & Keras
- PyTorch-TabNet
- RAPIDS (cuML, cuDF)
- CUDA, cuDNN
- AWS SageMaker

---

---

## 📈 Key Results Summary

- **XGBoost**: Training time reduced from 1859s (CPU) to 100s (GPU)
- **KMeans & KNN**: Training 4–30x faster on GPU; KNN inference still heavy on all platforms
- **TabNet**: Training 16x faster on GPU; consistent inference speed
- **MLP**: Modest GPU gains, fast inference
- **LSTM/BiLSTM**: Huge GPU training speedups; inference slightly slower due to small batch overhead

---

## 💡 Conclusion

-  **GPU offers significant gains for deep and complex models**, especially during training.
-  **Inference speed varies** — GPU only wins if batch size and model size justify the overhead.
-  **AWS GPU instances (p3.2xlarge)** are effective, but optimal only when matched with the right workload.
-  **For small or linear models, **CPU may offer better cost-performance trade-offs.**



## 📜 References

1. Çolhak et al., *GPU-Accelerated Machine Learning in IoV* — arXiv:2504.01905  
2. Gyawali, *CPU vs GPU for Deep Learning* — arXiv:2309.02521  
3. Pangre et al., *Hardware Benchmarks for DL* — IJETR 2020  
4. Kaggle Datasets, UCI Archive, AWS Documentation

> 📌 This repository helps understand which hardware works best for which model — empowering smarter, resource-aware ML deployment decisions.
