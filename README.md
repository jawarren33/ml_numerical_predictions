# ml_numerical_predictions
This is a core repository for storing frequently used models used in numerical data prediction models.

The FeatureSelection class uses the SelectKBest method from scikit-learn to perform feature selection based on the f_regression scoring function. It selects the top k_features based on the scores and stores them in the selected_features attribute.

The LinearRegressionModel class encapsulates the linear regression model using scikit-learn's LinearRegression. It provides methods to fit the model on the training data and make predictions on new data.

The test_linear_regression function takes the training and testing data, fits the linear regression model, predicts the target variable for the test data, and calculates the mean squared error (MSE) as a measure of accuracy.


1. Linear Regression: A linear regression model is used to establish a linear relationship between the input features and the target variable. It predicts a continuous numerical value based on the input variables.

2. Decision Trees: Decision trees are hierarchical models that make predictions by splitting the data based on a set of decision rules. They are capable of handling both numerical and categorical variables.

3. Random Forest: Random forest is an ensemble model that combines multiple decision trees to make predictions. It improves prediction accuracy by reducing overfitting and capturing the collective knowledge of multiple trees.

4. Gradient Boosting: Gradient boosting is another ensemble method that combines weak models, such as decision trees, in a sequential manner to create a stronger predictive model. It iteratively corrects the errors of the previous models and improves prediction accuracy.

5. Neural Networks: Neural networks, specifically deep learning models, are powerful for numerical prediction tasks. They consist of multiple layers of interconnected nodes (neurons) and can learn complex patterns from the data. Common architectures include feedforward neural networks, convolutional neural networks (CNNs), and recurrent neural networks (RNNs).

6. Support Vector Machines (SVM): SVM is a supervised learning model that separates data into different classes using hyperplanes. It can also be used for regression tasks, where it predicts a continuous value by finding the best-fit hyperplane.

7. K-Nearest Neighbors (KNN): KNN is a non-parametric model that predicts a value based on the k closest data points in the feature space. It is suitable for both classification and regression tasks.

8. XGBoost: XGBoost (Extreme Gradient Boosting) is an optimized implementation of gradient boosting that offers high performance and scalability. It is widely used in data science competitions and has become popular for various numerical prediction tasks.

9. Time Series Models: Time series models, such as ARIMA (Autoregressive Integrated Moving Average) and LSTM (Long Short-Term Memory), are specifically designed to handle time-dependent data and make predictions based on historical patterns and trends.

10. Gaussian Processes: Gaussian Processes model the target variable as a distribution over functions, allowing uncertainty estimation in predictions. They are effective when dealing with limited data or when capturing complex nonlinear relationships.
