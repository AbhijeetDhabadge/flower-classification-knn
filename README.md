# 🌸 Flower Classification using K-Nearest Neighbors (KNN)
Machine Learning Project | Python | Scikit-learn | Iris Dataset

A machine learning project that classifies iris flowers into three species — *setosa*, *versicolor*, and *virginica* — using the K-Nearest Neighbors (KNN) algorithm, based on four measurements: sepal length, sepal width, petal length, and petal width.

> BSc Artificial Intelligence and Machine Learning — First Year Project

## 📌 Overview

This project builds and evaluates a KNN classifier on the classic Iris dataset, following a complete ML pipeline:

1. **Load** the dataset
2. **Explore** it (summary stats, class distribution)
3. **Visualize** feature relationships and distributions
4. **Scale** features using `StandardScaler` (fit on training data only, to avoid data leakage)
5. **Tune K** using 5-fold cross-validation on the training set
6. **Train** the final model with the best K
7. **Evaluate** on a held-out test set (accuracy, classification report, confusion matrix)
8. **Predict** on new, unseen flower samples
9. **Analyze** feature correlation with the target

## 🧠 Why KNN?

KNN is a distance-based algorithm that classifies a sample based on the majority class among its K nearest neighbors in feature space. Because it relies on distances, feature scaling is essential — otherwise features on larger numeric scales would dominate the distance calculation.

## 📂 Project Structure

```
flower-classification-knn/
├── flower_classification_knn.ipynb   # Main notebook (full pipeline)
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Files/folders excluded from git
```

## ⚙️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/flower-classification-knn.git
   cd flower-classification-knn
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook flower_classification_knn.ipynb
   ```

## 📊 Dataset

The project uses the built-in **Iris dataset** from `sklearn.datasets`, so no external download is needed. It contains 150 samples, 50 per species, with 4 numeric features each.

## ✅ Results

- The model selects the best K value via cross-validation on the training set only, then reports final accuracy, a full classification report, and a confusion matrix on the held-out test set.
- Petal measurements separate the species better than sepal measurements — *setosa* is linearly separable, while *versicolor* and *virginica* overlap slightly.

## 🛠️ Tech Stack

- Python 3
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
