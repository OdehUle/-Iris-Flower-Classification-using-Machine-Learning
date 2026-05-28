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

# ⚙️ Technologies Used

| Tool         | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming               |
| NumPy        | Numerical Computing       |
| Pandas       | Data Manipulation         |
| Matplotlib   | Visualization             |
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

# 🖼️ Project Preview

## Iris Flower
<img width="640" height="448" alt="iris" src="https://github.com/user-attachments/assets/8bc47f43-f528-4ea5-9944-a25c8bb8c671" />

---

# 📊 Dataset Information

The dataset used is the classical Iris dataset available in Scikit-Learn.

```python
from sklearn.datasets import load_iris
iris_data= load_iris()
print('Keys of iris_dataset: \n{}'.format(iris_data.keys()))
print(iris_data['DESCR'][:193]+ '\n...')
```

## Dataset Statistics

| Feature       | Value                     |
| ------------- | ------------------------- |
| Total Samples | 150                       |
| Features      | 4                         |
| Classes       | 3                         |
| Problem Type  | Multiclass Classification |

---

## Feature Names
```python
print('Target names: {}'.format(iris_data['target_names']))
print('Feature names: \n{}'.format(iris_data['feature_names']))
```


| Features          |
| ----------------- |
| Sepal Length (cm) |
| Sepal Width (cm)  |
| Petal Length (cm) |
| Petal Width (cm)  |

---

# 🌺 Iris Flower Anatomy

<img width="640" height="448" alt="iris_2" src="https://github.com/user-attachments/assets/4d3aea86-6c66-402c-9d11-63853d1d463a" />

---

# 🔍 Exploratory Data Analysis

A scatter matrix was used to visualize feature relationships and class separability.

## Key Observations

* Setosa is clearly separable from the other species.
* Petal measurements provide strong predictive information.
* Versicolor and Virginica exhibit partial overlap.

---

# 📈 Scatter Matrix Visualization
```python
import pandas as pd
import numpy as np
import mglearn as mg
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

iris = pd.DataFrame(X_train, columns=iris_data.feature_names)

grr = scatter_matrix(
    iris,
    c=y_train,
    figsize=(15, 15),
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=0.8,
    cmap=mg.cm3
)

plt.show()
```

<img width="1229" height="1222" alt="scatter_matric" src="https://github.com/user-attachments/assets/9e271963-7433-4fcd-aabf-691152c0cfa2" />

The three classes seem to be relatively well separated using the sepal and petal measurements. This means that a machine learning model will likely be able to learn to separate them.


---



# 🔬 Model Training
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_data['data'], iris_data['target'], random_state=0)
print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('X_test shape: {}'.format(X_test.shape))
print('y_test shape: {}'.format(y_test.shape))
```
The dataset was divided into:

* **75% Training Data**
* **25% Testing Data**

The model was trained using supervised learning techniques for multiclass classification.

---

# 📈 Model Performance

```python
print('Test set score: {:.2f}'.format(knn.score(X_test, y_test)))
```

| Metric       | Result |
| ------------ | ------ |
| Accuracy     | 97%+   |
| Classes      | 3      |
| Test Samples | 38     |

The trained model achieved high accuracy on unseen test data.

---

# 🚀 Sample Prediction

```python
prediction = knn.predict(X_new)
print('Prediction: {}'.format(prediction))
print('Predicted target name: {}'.format(iris_data['target_names'][prediction]))
```

### Output

```python
Prediction: [0]
Predicted target name: ['setosa']
```

---

# 🧪 Test the Model with New Data

Let us use the following unseen flower measurements to test the model:

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
predictions = knn.predict(new_samples)

for sample, pred in zip(new_samples, predictions):
    print(f"{sample} → {iris_data.target_names[pred]}")
```

---

### Output

```python
[5.0, 3.4, 1.5, 0.2] → setosa
[6.4, 3.2, 4.5, 1.5] → versicolor
[6.9, 3.1, 5.4, 2.1] → virginica
[5.5, 2.6, 4.4, 1.2] → versicolor
[7.2, 3.6, 6.1, 2.5] → virginica
[4.8, 3.0, 1.4, 0.1] → setosa
[6.0, 2.9, 4.5, 1.5] → versicolor
[6.7, 3.3, 5.7, 2.4] → virginica
[5.1, 3.8, 1.6, 0.2] → setosa
[6.3, 2.7, 4.9, 1.8] → virginica
```
---

# 📂 Project Structure

```bash
iris-flower-classification/
│
├── data/
├── images/
│   ├── iris.png
│   ├── iris_2.png
│   └── scatter-matrix.png
│
├── notebooks/
│   └── iris_classification.ipynb
│
├── requirements.txt
├── README.md
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
git clone [https://github.com/OdehUle/iris-flower-classification.git](https://github.com/OdehUle/-Iris-Flower-Classification-using-Machine-Learning
```

Move into the project directory:

```bash
cd Iris-Flower-Classification-using-Machine-Learning
```
---
# 🤝 Acknowledgement

The Iris dataset is one of the most well-known datasets in Machine Learning and Statistics, introduced by Ronald A. Fisher.

---
# ⭐ Support

If you found this project helpful, consider giving it a star ⭐ on GitHub.

```
```
