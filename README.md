# ML-Exercise-Income-Dataset-Regression-Metrics

This repository contains an exercise on regression metrics using an income dataset to predict happiness. The exercise includes data preprocessing, model training, evaluation, and visualization.   

![image](https://github.com/user-attachments/assets/a5b61eb6-3238-436d-9872-fd8b0cef69fa)

## Overview

- **Coded by**: Himel Sarder
- **Contact**: info.himelcse@gmail.com
- **LinkedIn**: [Himel Sarder](https://www.linkedin.com/in/himel-sarder/)

## Files in the Repository

- `Exercise ~ Regression Metrics.ipynb`: Jupyter notebook containing the regression analysis.
- `LICENSE`: License information.
- `Mymodel.pkl`: Serialized model file.
- `README.md`: This README file.
- `Regression Metrics.ipynb`: Additional notebook for regression metrics.
- `income.csv`: Dataset containing income and happiness data.

## Dataset

The dataset `income.csv` contains the following columns:
- `Unnamed: 0`: Index column.
- `income`: Income values.
- `happiness`: Happiness scores.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Himel-Sarder/ML-Exercise-Income-Dataset-Regression-Metrics.git
   ```
2. Navigate to the project directory:
   ```sh
   cd ML-Exercise-Income-Dataset-Regression-Metrics
   ```
3. Install the required libraries:
   ```sh
   pip install pandas numpy matplotlib scikit-learn
   ```

## Usage

### 1. Load and Explore the Dataset
Load the dataset using pandas and display its structure:
```python
import pandas as pd
df = pd.read_csv('income.csv')
print(df.head())
print(df.shape)
print(df.info())
```

### 2. Data Visualization
Visualize the relationship between income and happiness:
```python
import matplotlib.pyplot as plt
plt.scatter(df['income'], df['happiness'], c=df['happiness'], cmap='coolwarm')
plt.xlabel('Income')
plt.ylabel('Happiness')
plt.colorbar(label='Happiness')
plt.show()
```

### 3. Data Splitting
Split the data into training and test sets:
```python
from sklearn.model_selection import train_test_split
X = df.iloc[:, 1:2]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```

### 4. Model Training
Train a Linear Regression model:
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
```

### 5. Model Evaluation
Evaluate the model using various metrics:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = lr.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
```

### 6. Save the Model
Save the trained model to a file:
```python
import pickle
pickle.dump(lr, open('Mymodel.pkl', 'wb'))
```

## Additional Experiment
Test the impact of adding random features and recalculating R² and adjusted R² scores.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- Thank you to everyone who contributed to this project.

## Contact

If you have any questions or feedback, feel free to contact me at info.himelcse@gmail.com.

---

Happy coding! 😺
