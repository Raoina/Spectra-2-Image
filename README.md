# Spectra-to-Image Deep Learning Project

## 📌 Overview

This project, developed under **Sci-Ware**, explores **spectral data transformation** into **2D image representations** and applies **deep learning models (CNNs)** for regression tasks such as predicting **Moisture** and other chemical properties.
Additionally, a **PLS (Partial Least Squares)** baseline model is implemented for comparison with traditional chemometrics.

---

## 🎯 Objectives

* Convert **1D spectral data** into structured **2D images** using different reshaping techniques:

  * **Raw Reshape (Row-major order)**
  * **Column-major Reshape**
  * **Snake Pattern Reshape**
* Apply preprocessing methods:

  * Transmission → Absorbance
  * Derivative transformation (Savitzky-Golay, 2nd order)
  * Mean centering / Standard Normalization (Z-score)
  * Resampling spectra to fixed dimensions
* Build deep learning pipelines:

  * **2D CNN model** (inspired by VGG-like architecture)
  * Compare with **PLS baseline** using SIMPLS algorithm
* Evaluate performance using:

  * RMSEC, RMSECV, RMSEP
  * R² (Calibration, Cross-validation, Prediction)
  * Bias analysis

---

## 🧪 Dataset

* **X-block (Spectra):**

  * Samples: 432
  * Variables: 257 spectral points
  * Preprocessing: Transmission → Absorbance, 2nd derivative, mean centering

* **Y-block (Targets):**

  * Example target: **Moisture (%)**
  * Format: CSV file with one column per property

⚠️ Dataset and experimental data are proprietary and belong to **Sci-Ware**.

---

## 🔄 Data Transformation

Spectral data is resampled and reshaped into **2D images**:

* **Snake Pattern Example:**

```python
def to_snake_pattern(vector, img_size):
    img = np.zeros((img_size, img_size))
    for i in range(img_size):
        row = vector[i * img_size:(i + 1) * img_size]
        if i % 2 == 1:
            row = row[::-1]
        img[i, :] = row
    return img
```

* Final image size used: **65 × 65** pixels

---

## 🧠 Models

### 🔹 1. PLS Regression (Baseline)

* Algorithm: **SIMPLS**
* Latent Variables: 8
* Cross-validation: Venetian blinds (5 splits)
* Example results for Moisture:

  * RMSEC = 2.1651
  * RMSECV = 2.2557
  * RMSEP = 1845.59
  * R² Cal = 0.8876
  * R² CV = 0.8780
  * R² Pred = 0.9050

---

### 🔹 2. CNN Model (Deep Learning)

Input: **1 × 65 × 65 image**

Architecture:

```
Input → Conv2D(64,3×3) → MaxPool(2×2)  
      → Conv2D(128,3×3) → MaxPool(2×2)  
      → Conv2D(256,3×3) → Conv2D(256,3×3) → MaxPool(2×2)  
      → Conv2D(512,3×3) → Conv2D(512,3×3) → MaxPool(2×2)  
      → Conv2D(512,3×3) → Conv2D(512,3×3) → MaxPool(2×2)  
      → Flatten → Dense(128) → Dense(Output)
```

Implemented in **PyTorch**.

---

## ⚖️ Evaluation

* Comparison between **PLS** (classical regression) vs **CNN** (deep learning).
* CNN showed higher prediction accuracy on transformed spectral images compared to PLS in cross-validation.
* Remaining work: Extend CNN evaluation on all transformation methods (Raw, Column, Snake).

---

## 🛠️ Technologies

* Python 3.10
* Libraries:

  * **PyTorch** (deep learning)
  * **Scikit-learn** (PLS, metrics)
  * **SciPy** (resampling)
  * **NumPy, Pandas** (data handling)
  * **Matplotlib/Seaborn** (visualization)

---

## 📊 Example Visualizations

* Original spectra
* Resampled spectra
* 2D image (snake pattern)

*(Add plots here if available)*

---

## 🚀 Future Work

* Compare image reshaping techniques (Row-major vs Column-major vs Snake).
* Test CNN on larger datasets.
* Experiment with transfer learning from pretrained vision models.
* Optimize hyperparameters (batch size, learning rate, dropout).

---

## 👨‍💻 Author & Ownership

* Developed by **Rowaina Reda** and **Salma Bassem**
* 📅 Started: Sept 2025
* 🏢 Project under **Sci-Ware**

---
