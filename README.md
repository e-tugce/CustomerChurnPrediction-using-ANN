# Customer Churn Prediction using Artificial Neural Network
This project aims to predict customer churn using an Artificial Neural Network (ANN). Customer churn refers to the phenomenon where customers stop doing business with a company. In this project, we utilize customer features from a dataset to create and train an ANN model. The output of the model predicts the probability of a customer churning.

## Dataset
The dataset used in this project is the "Churn_Modelling.csv" file. It contains various features of customers such as credit score, geography, gender, age, tenure, balance, etc. The target variable is 'Exited', indicating whether a customer churned or not.

### Implementation
The dataset is loaded using pandas.
Features and target variable are determined.
Categorical variables are encoded using LabelEncoder and OneHotEncoder.
The data is split into training and testing sets.
Standard scaling is applied to the data.
An ANN model is constructed using TensorFlow's Keras API.
The model is compiled with appropriate loss and optimization functions.
The model is trained on the training data.
Model performance is evaluated using metrics such as accuracy, confusion matrix, and classification report.
