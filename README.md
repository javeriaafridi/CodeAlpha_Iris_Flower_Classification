# CodeAlpha_Iris_Flower_Classification

**CodeAlpha Data Science Internship — Task 1**

## Objective
Classify Iris flowers into one of three species (*setosa*, *versicolor*, *virginica*) using sepal and petal measurements.

## Dataset
`Iris.csv` — 150 samples, 4 numeric features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`), balanced across 3 classes (50 each).

## Approach
1. **EDA** — pairplot and correlation heatmap to understand feature relationships.
2. **Preprocessing** — label-encoded the target, standardized features with `StandardScaler`.
3. **Modeling** — trained and compared 5 classifiers:
   - Logistic Regression
   - K-Nearest Neighbors
   - Decision Tree
   - Random Forest
   - SVM (RBF kernel)
4. **Evaluation** — accuracy, precision/recall/F1 per class, confusion matrix.

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 93.3% |
| K-Nearest Neighbors | 93.3% |
| Decision Tree | 90.0% |
| Random Forest | 90.0% |
| **SVM (RBF kernel)** | **96.7%** |

The **SVM (RBF kernel)** performed best on the held-out test set.

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python iris_classification.py
```
Plots are saved to `outputs/`.

## Files
- `iris_classification.py` — full pipeline (EDA, preprocessing, training, evaluation)
- `Iris.csv` — dataset
- `outputs/` — generated plots (pairplot, correlation heatmap, confusion matrix, model comparison)

---
*Submitted as part of the CodeAlpha Data Science Internship.*
