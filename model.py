'''
This Python software uses the diabetes dataset to train a number of classification models.
Using pandas, the dataset is loaded, and some fundamental data analysis is performed, such as
 looking for missing values and displaying the distribution of each characteristic. '''
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# https://numpy.org/doc/stable/reference/
# https://www.jcchouinard.com/confusion-matrix-in-scikit-learn/
import pandas as pd
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import rfc as rfc
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# load the diabetes dataset
diabetes = pd.read_csv("diabetes.csv")

# data attribute information
print(diabetes.info())

# to observe the shape of the dataframe.
print(diabetes.shape)

# checking missing value
print(diabetes.isnull().sum())

# -----------------------------
corr=diabetes.corr()
sns.heatmap(corr, annot=True) # an array of the same shape as data which is used to annotate the heatmap

# -----------------------

# ------------------------------------
# loop through all numeric columns except for the target column
for col in diabetes.columns[0:-1]:
    # plot the distribution of the current column using seaborn's displot function
    # set the dataframe to diabetes_df using the data parameter
    # set the x-axis to the current column being iterated using the x parameter
    # set kde parameter to True to display kernel density estimate along with the histogram
    sns.displot(data=diabetes, x=col, kde=True)

# ---
# split data into features and target variable
X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----
# checking imbalance using oversampling and under sampling using SMOTE and RandomUnderSampler
smote = SMOTE()
rus = RandomUnderSampler()
x_resample, y_resample = smote.fit_resample(diabetes.iloc[:, :-1], diabetes.iloc[:, -1])
x_resample, y_resample = rus.fit_resample(x_resample, y_resample)

# testing is balanced now
print(y_resample.value_counts())

# data needs to normalize
scaler = StandardScaler()
x_resample = scaler.fit_transform(x_resample)

# split data into features and target variable
X = x_resample
y = y_resample

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert the scaled training data into a dataframe
X_train_df = pd.DataFrame(X_train, columns=diabetes.columns[:-1])

# create a list of classifiers
classifiers = [
    DecisionTreeClassifier(random_state=42),
    LogisticRegression(random_state=42),
    KNeighborsClassifier(),
    SVC(random_state=42),
    GaussianNB(),
    GradientBoostingClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
]


# define a function to train and evaluate each classifier
def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(classifier)
    print(f"Accuracy: {round(accuracy, 2)}")
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"F1-score: {round(f1, 2)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


# evaluate the Decision Tree classifier
classifier = RandomForestClassifier(random_state=42)
evaluate_classifier(classifier, X_train, y_train, X_test, y_test)

# loop through each classifier in the list of classifiers and train and evaluate them
for classifier in classifiers:
    evaluate_classifier(classifier, X_train, y_train, X_test, y_test)

# create a dictionary to store the results
results = {}


# define a function to train and evaluate each classifier and return the results as a dictionary
def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    results = {'accuracy': accuracy,
               'precision': precision,
               'recall': recall,
               'f1_score': f1,
               'confusion_matrix': cm,
               'classification_report': cr}
    return results

# -------------------------------------
# import required libraries
from sklearn.metrics import roc_auc_score, roc_curve

# loop through each classifier in the list of classifiers and calculate ROC Curve
for classifier in classifiers:
    # fit the classifier
    classifier.fit(X_train, y_train)
    # check if predict_proba is available
    if hasattr(classifier, 'predict_proba'):
        # predict probabilities for the positive class
        y_prob = classifier.predict_proba(X_test)[:,1]
    else:
        # predict scores for the positive class using decision_function or predict
        y_prob = classifier.decision_function(X_test)
        # if using predict method instead
        # y_prob = classifier.predict(X_test)
    # calculate ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    # plot ROC Curve
    plt.plot(fpr, tpr, label=f'{classifier.__class__.__name__} (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    # plot a diagonal line representing the random guess classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    # set plot title, x-axis label, y-axis label, and legend
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {diabetes}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    # display the plot
    plt.show()


# --------------------------------------
# loop through each classifier in the list of classifiers and train and evaluate them
for classifier in classifiers:
    results[classifier.__class__.__name__] = evaluate_classifier(classifier, X_train, y_train, X_test, y_test)

# create a copy of the diabetes dataset
diabetes_copy = diabetes.copy()

# add a new column to the copy indicating whether a patient has diabetes or not
diabetes_copy["has_diabetes"] = np.where(diabetes_copy["Outcome"] == 1, "yes", "no")

# plot the distribution of patients with and without diabetes
sns.countplot(x="has_diabetes", data=diabetes_copy)
plt.title("Distribution of Patients with and Without Diabetes")
plt.xlabel("Has Diabetes")
plt.ylabel("Count")
plt.show()

# plot the distribution of patients with and without diabetes
sns.countplot(x="has_diabetes", hue="Pregnancies", data=diabetes_copy)
plt.title("Distribution of Patients with and Without Diabetes")
plt.xlabel("Has Diabetes")
plt.ylabel("Count")
plt.show()

# create a figure
fig = plt.figure(figsize=(10, 10))

# create a subplot for accuracy
plt.subplot(2, 2, 1)
plt.bar(results.keys(), [result['accuracy'] for result in results.values()])
plt.title("Accuracy")

# create a subplot for precision
plt.subplot(2, 2, 2)
plt.bar(results.keys(), [result['precision'] for result in results.values()])
plt.title("Precision")

# create a subplot for recall
plt.subplot(2, 2, 3)
plt.bar(results.keys(), [result['recall'] for result in results.values()])
plt.title("Recall")

# create a subplot for f1-score
plt.subplot(2, 2, 4)
plt.bar(results.keys(), [result['f1_score'] for result in results.values()])
plt.title("F1-score")

# display the figure
plt.show()

# ----

# loop through each classifier in the list of classifiers and train and evaluate them
for classifier in classifiers:
    clf_name = type(classifier).__name__
    clf_scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
    results[clf_name] = clf_scores

# create a dataframe from the dictionary of results
results_df = pd.DataFrame.from_dict(results)

# plot the results
sns.boxplot(data=results_df)
plt.title('Classifier Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()
# ----

# -------------------------

# fit the Random Forest classifier on the entire dataset
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# create a feature importance plot using seaborn's bar plot function
sns.barplot(x=rfc.feature_importances_, y=diabetes.columns[:-1])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# -----------------------------------

# Create a list to store the accuracy rates
accuracy_rate = []

# Create a range of n_estimators to test
estimators = range(1, 25)

# Loop through different n_estimators
for n in estimators:
    # Create a random forest classifier model
    rfc = RandomForestClassifier(n_estimators=n, random_state=42)
    # Train the model on the training set
    rfc.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = rfc.predict(X_test)
    # Compute the accuracy of the model and append to the list
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_rate.append(accuracy)

# Plot the accuracy rate
plt.plot(estimators, accuracy_rate, color='blue', linestyle='dashed', marker='o')
plt.title('Accuracy Rate vs. range of estimators')
plt.xlabel('Range of estimaters')
plt.ylabel('Accuracy Rate')
plt.show()


# -----------------------------------------------
# Code to plot accuracy against that k
plt.figure(figsize=(10,6))

plt.plot(range(1,25),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K-Value')
plt.ylabel('Accuracy Rate')
# --------------------

# ---------------------------------------------
# define range of estimators to test
estimator_range = range(1, 25)

# calculate accuracy rate for different number of estimators
accuracy_rate = []
for n_estimators in estimator_range:
    rfc = RandomForestClassifier(n_estimators=n_estimators)
    score = cross_val_score(rfc, X, y, cv=8)
    accuracy_rate.append(score.mean())

# plot accuracy rate
plt.plot(estimator_range, accuracy_rate)
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy Rate')
plt.title('RandomForestClassifier Accuracy Rate vs. Number of Estimators')
plt.show()
# ---------------------------------------
# make a pickle file model
pickle.dump(classifier, open("model.pkl", "wb"))
