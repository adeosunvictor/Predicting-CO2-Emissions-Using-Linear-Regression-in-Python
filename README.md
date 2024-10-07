# Predicting-CO2-Emissions-Using-Linear-Regression-in-Python

Here's a professionally written article based on your code that you can use for your GitHub repository:

---

 Predicting CO2 Emissions Using Linear Regression in Python

In this project, we’ll use machine learning techniques, specifically **Linear Regression**, to predict CO2 emissions based on various attributes such as engine size, number of cylinders, and fuel consumption. This article will guide you through loading a dataset, visualizing relationships between variables, splitting the data, training the model, and evaluating its performance.

it is better to run the code on google colab or kaggle 

 Tools and Libraries

To begin, we need to install and import the necessary Python libraries:

```bash
!pip install scikit-learn
!pip install matplotlib
!pip install pandas
!pip install numpy
```

Next, we import the libraries that will be used in our analysis:

```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

 Dataset

We’ll be using a dataset called **FuelConsumption.csv**, which contains information on various car attributes and their respective CO2 emissions.

```python
 Load the data
df = pd.read_csv("FuelConsumption.csv")
```

To understand the structure of the dataset, let’s select a few columns of interest:

```python
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(11)
```

 Visualizing the Data

Before building the model, it’s helpful to visualize the relationship between engine size and CO2 emissions. This can give us a preliminary idea of how well linear regression might perform.

```python
 Plot the data
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='red')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
```

As we can see from the scatter plot, there seems to be a positive correlation between engine size and CO2 emissions. Now, let's move on to building the predictive model.

 Preparing the Data

We'll select `ENGINESIZE`, `CYLINDERS`, and `FUELCONSUMPTION_COMB` as our feature variables (independent variables), and `CO2EMISSIONS` as our target variable (dependent variable).

```python
 Define the features (X) and target (y)
x = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y = df[['CO2EMISSIONS']]
```

 Splitting the Data

Next, we split the dataset into training and test sets. This allows us to train the model on one subset of the data and evaluate its performance on unseen data.

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```

 Building and Training the Model

We will use **Linear Regression** from the `scikit-learn` library to build our model. Linear regression attempts to model the relationship between two or more variables by fitting a linear equation to observed data.

```python
from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)
```

 Making Predictions

After training the model, we can use it to make predictions on the test data:

```python
 Make predictions
y_pred = model.predict(x_test)
print(y_pred)
```

 Evaluating the Model

To evaluate the model's performance, we’ll use **Mean Squared Error (MSE)** and **R-squared (R²)**. MSE measures the average squared difference between actual and predicted values, while R² explains the proportion of variance in the dependent variable that is predictable from the independent variables.

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

Custom Prediction

You can also predict CO2 emissions for custom inputs, using the same trained model:

```python
 Example for custom prediction (values should be structured like the input features)
 custom_values = [[engine_size, cylinders, fuel_consumption_comb]]
 predicted_emission = model.predict(custom_values)
```

 Conclusion

In this project, we successfully built a simple linear regression model to predict CO2 emissions from various vehicle attributes. The model is evaluated using metrics like MSE and R² to gauge its performance. This approach can be further extended with more complex models and additional features to improve prediction accuracy.

 Future Work

1. Feature Engineering: Additional vehicle features such as transmission type, fuel type, or vehicle weight can be added to improve model accuracy.
2. Model Tuning: Hyperparameters of the linear regression model can be fine-tuned or even replaced by more advanced models like Ridge or Lasso regression.
3. Cross-validation: To ensure the robustness of our results, cross-validation techniques can be employed.

