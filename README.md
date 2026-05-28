# 🌸 Iris Flower Classification using Machine Learning

> My first end-to-end Machine Learning project for multiclass classification using the famous Iris dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

# 🌼 Project Overview

This project builds a Machine Learning model capable of classifying iris flowers into three species:

- **Setosa**
- **Versicolor**
- **Virginica**

The model learns from four flower measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The objective is to predict the species of a new iris flower based on these measurements.

---

# 🖼️ Project Preview

## Iris Flower
![Iris Flower](images/iris.png)


---

# 📊 Dataset Information

The dataset used is the classical Iris dataset available in Scikit-Learn.

## Dataset Statistics

| Feature       | Value                     |
| ------------- | ------------------------- |
| Total Samples | 150                       |
| Features      | 4                         |
| Classes       | 3                         |
| Problem Type  | Multiclass Classification |

---

## Feature Names

| Features          |
| ----------------- |
| Sepal Length (cm) |
| Sepal Width (cm)  |
| Petal Length (cm) |
| Petal Width (cm)  |

---

# 🌺 Iris Flower Anatomy

> Add the iris flower anatomy graphic below.

```md
![Iris Anatomy](images/iris-parts.png)
```

---

# 🔍 Exploratory Data Analysis

A scatter matrix was used to visualize feature relationships and class separability.

## Key Observations

* Setosa is clearly separable from the other species.
* Petal measurements provide strong predictive information.
* Versicolor and Virginica exhibit partial overlap.

---

# 📈 Scatter Matrix Visualization

> Add your scatter matrix visualization here.

```md
![Scatter Matrix](images/scatter-matrix.png)
```

---

# ⚙️ Technologies Used

| Tool         | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming               |
| NumPy        | Numerical Computing       |
| Pandas       | Data Manipulation         |
| Matplotlib   | Visualization             |
| Seaborn      | Statistical Visualization |
| Scikit-Learn | Machine Learning          |

---

# 🧠 Machine Learning Workflow

```text
Load Dataset
     ↓
Explore Data
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Prediction
     ↓
Evaluation
```

---

# 🔬 Model Training

The dataset was divided into:

* **75% Training Data**
* **25% Testing Data**

The model was trained using supervised learning techniques for multiclass classification.

---

# 📈 Model Performance

| Metric       | Result |
| ------------ | ------ |
| Accuracy     | 97%+   |
| Classes      | 3      |
| Test Samples | 38     |

The trained model achieved high accuracy on unseen test data.

---

# 🚀 Sample Prediction

```python
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
print(prediction)
```

### Output

```python
Setosa
```

---

# 📂 Project Structure

```bash
iris-flower-classification/
│
├── data/
├── images/
│   ├── iris-banner.png
│   ├── iris-parts.png
│   └── scatter-matrix.png
│
├── notebooks/
│   └── iris_classification.ipynb
│
├── requirements.txt
├── README.md
└── model.pkl
```

---

# 💡 Key Concepts Demonstrated

* Supervised Learning
* Classification
* Train/Test Split
* Data Visualization
* Predictive Modelling
* Model Evaluation

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/iris-flower-classification.git
```

Move into the project directory:

```bash
cd iris-flower-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

# 📸 Recommended Additional Visualizations

You can improve this project further by adding:

* ✅ Confusion Matrix
* ✅ Pairplot
* ✅ Heatmap
* ✅ Decision Boundary Plot
* ✅ Feature Importance Plot

---

# 🌟 Future Improvements

* Deploy with Streamlit
* Add real-time prediction interface
* Hyperparameter tuning
* Cross-validation experiments
* Save trained model with Pickle

---

# 🤝 Acknowledgement

The Iris dataset is one of the most well-known datasets in Machine Learning and Statistics, introduced by Ronald A. Fisher.

---

# 🧪 Test the Model with New Data

Use the following unseen flower measurements to test the model:

```python
new_samples = [
    [5.0, 3.4, 1.5, 0.2],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 5.4, 2.1],
    [5.5, 2.6, 4.4, 1.2],
    [7.2, 3.6, 6.1, 2.5],
    [4.8, 3.0, 1.4, 0.1],
    [6.0, 2.9, 4.5, 1.5],
    [6.7, 3.3, 5.7, 2.4],
    [5.1, 3.8, 1.6, 0.2],
    [6.3, 2.7, 4.9, 1.8]
]
```

## Predict New Samples

```python
predictions = model.predict(new_samples)

for sample, pred in zip(new_samples, predictions):
    print(f"{sample} → {iris_dataset.target_names[pred]}")
```

---

# ⭐ Support

If you found this project helpful, consider giving it a star ⭐ on GitHub.

```
```
