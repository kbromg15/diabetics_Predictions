# diabetics_predictions
Diabetes Prediction Project This project leverages machine learning models to predict the likelihood of diabetes in individuals based on clinical parameters. 
It utilizes the Pima Indians Diabetes Database to train and evaluate the models. A web application is included for users to input their data and obtain predictions seamlessly.

Project Structure

• app.py: Flask web application for hosting the user interface and generating predictions using the trained model.

• model.py: Script for data preprocessing, model training, evaluation, and saving the trained model as a serialized file.

• diabetes.csv: Dataset used for training and testing the machine learning models.

• model.pkl: Pickle file containing the serialized version of the trained model for deployment.

• templates/index.html: Frontend template for the web application, allowing users to input data.


Get Predictions:

• Input clinical data such as glucose level, blood pressure, and other features to receive the diabetes prediction.

Machine Learning Models Implemented

• Logistic Regression

• Decision Tree Classifier

• K-Nearest Neighbors Classifier

• Support Vector Classifier

• Gaussian Naive Bayes

• Gradient Boosting Classifier

• Random Forest Classifier


Evaluation Metrics

The models are evaluated using the following metrics:

• Accuracy

• Precision

• Recall

• F1 Score

• Confusion Matrix


Results

• The Random Forest Classifier demonstrated the best performance, achieving the highest accuracy among the models tested.

