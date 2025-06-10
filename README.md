# Diabetes

# 🤖 Design of a Low-Complexity Deep Learning Model for Diagnosis of Type 2 Diabetes

**Author**: [Soroush Soltanizadeh](https://www.linkedin.com/in/soroush-soltanizadeh-1136892b6/)
**Google Scholar**: [Profile](https://scholar.google.com/citations?user=ARKNJYwAAAAJ&hl=en)

---

## 📌 Overview

Diabetes is a life-threatening chronic condition with the potential to cause heart disease, nerve damage, and organ failure. Early and efficient diagnosis of **Type 2 Diabetes** is essential for preventing complications. This project introduces a **low-complexity deep learning model** a 1D Convolutional Neural Network (CNN) that offers high accuracy and can be embedded in **wearable devices** or **IoT-based health monitoring systems**.

---

## 🧪 Research Objective

The main goal of this project is to balance **model accuracy** and **computational complexity**. We conduct an **accuracy-complexity trade-off study** to design a CNN-based model that is lightweight yet highly effective for real-world diabetes diagnosis applications.

---

## 📁 Dataset

* **Name**: PIMA Indian Diabetes Dataset (PIDD)
* **Source**: Publicly available through [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **Target**: `Outcome` (1 = Diabetic, 0 = Non-diabetic)
* **Features**: 8 clinical parameters including:

  * Pregnancies
  * Glucose
  * BloodPressure
  * SkinThickness
  * Insulin
  * BMI
  * DiabetesPedigreeFunction
  * Age

---

## 🧠 Model Architecture

### CNN Model Summary:

* **Conv1D Layer**: 128 filters, kernel size = 5, padding = "same"
* **MaxPooling Layer**: pool size = 3
* **Flatten Layer**
* **Dropout Layer**: rate = 0.05
* **Dense Layer**: 1 neuron, sigmoid activation

```python
Model Summary:
Input Shape   → (8, 1)
Conv1D        → filters=128, kernel=5
MaxPooling1D  → pool_size=3
Dropout       → rate=0.05
Dense         → units=1, activation='sigmoid'
```

---

## ⚙️ Training Details

* **Optimizer**: Adam (`learning_rate=0.01`)
* **Loss Function**: Binary Crossentropy
* **Evaluation**: 10-fold Cross-Validation
* **Early Stopping**: Patience = 150 epochs
* **Batch Size**: 64
* **Epochs**: 200 (with early stopping)

---

## 📊 Results

| Metric                     | Value                                                                             |
| -------------------------- | --------------------------------------------------------------------------------- |
| **Mean Accuracy**          | **93.89%**                                                                        |
| **Standard Deviation**     | \~ small                                                                          |
| **Model Complexity (Ops)** | Computed as:<br>`ni * nf * nk * (ns + 2*npad - dilation*(nk-1) - 1 / stride + 1)` |
| **Complexity (CCNN)**      | ≈ *(Computed Value)*                                                              |

> 🧠 The model balances precision and low complexity, making it ideal for **real-time, edge, and embedded systems**.

---

## 🔁 Accuracy vs. Complexity Study

An empirical study was conducted to compare various deep learning models, including:

* CNN
* MLP
* CNN + MLP Hybrid

The CNN+MLP hybrid model yielded the highest performance with an accuracy of **93.89%**, outperforming all other models tested.

---

## 🧬 Applications

* **Wearable Health Devices**
* **Mobile Diagnostic Apps**
* **IoT-Based Patient Monitoring Systems**
* **Embedded AI in Smart Clinics**

---

## 🚀 Getting Started

### ✅ Prerequisites

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### ▶️ Run the Model

```bash
python diabetes_diagnosis_cnn.py
```

---

## 📚 Future Improvements

* Extend the model to multi-disease detection
* Integrate SHAP or LIME for interpretability
* Deploy the model in an Android/iOS app
* Real-time API deployment for clinical use

---

## 📜 License

This project is licensed under the **MIT License**. You're free to use, modify, and distribute it.
