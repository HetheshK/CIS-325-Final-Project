import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt 
        
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#importing our dataset
df = pd.read_csv('study_performance.csv')

#This whole section is for learning about our dataset
print(df.head)
print(df.shape)


    #identifies which columns are numerical and which are categorical
numericColumn = [column for column in df.columns if df[column].dtype != 'O']
categoricalColumn = [column for column in df.columns if df[column].dtype == 'O']
numberNumeric = len(numericColumn)
numberCategorical = len(categoricalColumn)


print(f"The number of numerical columns are{numberNumeric} and the columns are {numericColumn}")
print(f"The number of categorical columns are{numberCategorical} and the columns are {categoricalColumn}")

    # gets the unique values of each categorical column
uniqueGender = df['gender'].unique()
print(f"The categories in gender are {uniqueGender}")


uniqueRace= df['race_ethnicity'].unique()
print(f"The categories in race are{uniqueRace}")

uniqueEducation = df['parental_level_of_education'].unique()
print(f"the categories in a parents education level are {uniqueEducation}")

uniqueLunch = df['lunch'].unique()
print(f"The categories in the lunch variable are {uniqueLunch}")

uniquePreparation = df['test_preparation_course'].unique()
print(f"the categories in test preperation variable are {uniquePreparation}")

    #Breakdown of data, function counts how many of each group are in a categorical column  
genderBreakdown = df['gender'].value_counts()
for value, count in genderBreakdown.items():
    print(f'Count of {value}: {count}')
    
raceBreakdown = df['race_ethnicity'].value_counts()
for value, count in raceBreakdown.items():
    print(f'Count of {value}: {count}')
    
educationBreakdown = df['parental_level_of_education'].value_counts()
for value, count in educationBreakdown.items():
    print(f'Count of {value}: {count}')
    
lunchBreakdown = df['lunch'].value_counts()
for value, count in lunchBreakdown.items():
    print(f'Count of {value}: {count}')
    
preparationBreakdown = df['test_preparation_course'].value_counts()
for value, count in preparationBreakdown.items():
    print(f'Count of {value}: {count}')
    
    #numerical columns min max avg
def get_min_max_avg(column_name):
    minVal = df[column_name].min()
    maxVal = df[column_name].max()
    avgVal = df[column_name].mean()
    return minVal,maxVal,avgVal

minMath, maxMath, avgMath = get_min_max_avg('math_score')
print(f'minimun value: {minMath}\nmaximum value:{maxMath}\navergae value:{avgMath}')

minReading, maxReading, avgReading = get_min_max_avg('reading_score')
print(f'minimun value: {minReading}\nmaximum value:{maxReading}\navergae value:{avgReading}')

minWriting, maxWriting, avgWriting = get_min_max_avg('writing_score')
print(f'minimun value: {minWriting}\nmaximum value:{maxWriting}\navergae value:{avgWriting}')


#transforming our dataset to add average column
df['averageScore'] = (df['math_score'] + df['reading_score'] + df['writing_score'])/3

print(df['averageScore'].mean)

#Plotting
sns.pairplot(df,hue='lunch')
sns.pairplot(df,hue='test_preparation_course')

#Predicting reading score
    #preparing x and y variables
X = df[['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course','math_score']]
y = df['reading_score']

    #encoding
X_encoded = pd.get_dummies(X, columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'], drop_first=True)
y_encoded = pd.get_dummies(y, drop_first=True)
    #Data split 80 train 20 test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.1, random_state=9)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=9)
rf_regressor.fit(X_train, y_train)

    # Predict on the test set
y_pred = rf_regressor.predict(X_test)

    # test accuracy
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

new_data = pd.DataFrame({
    'math_score': [85],
    'gender_male': [0],
    'race_ethnicity_group B': [1],
    'race_ethnicity_group C': [0],
    'race_ethnicity_group D': [0],
    'race_ethnicity_group E': [0],
    "parental_level_of_education_bachelor's degree": [0],
    'parental_level_of_education_high school': [1],
    "parental_level_of_education_master's degree": [0],
    'parental_level_of_education_some college': [0],
    'parental_level_of_education_some high school': [0],
    'lunch_standard': [1],
    'test_preparation_course_none': [0]
})
new_data_encoded = pd.get_dummies(new_data, drop_first=True)
predicted_average_score = rf_regressor.predict(new_data_encoded)
print(predicted_average_score)


    #plotting
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (Random Forest Regression)')
plt.show()
    #2nd plot
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 8))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), X_encoded.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance Plot (Random Forest)')
plt.show()




#predicting lunch type with every other data point
X_lunch = df.drop('lunch', axis=1) 
y_lunch = df['lunch']

X_encoded_lunch = pd.get_dummies(X_lunch, drop_first=True)
y_encoded_lunch = pd.get_dummies(y_lunch, drop_first=True)
X_train_lunch, X_test_lunch, y_train_lunch, y_test_lunch = train_test_split(X_encoded_lunch, y_encoded_lunch, test_size=0.1, random_state=40)
rf_classifier_lunch = RandomForestClassifier(n_estimators=100, random_state=40)
rf_classifier_lunch.fit(X_train_lunch, y_train_lunch)
y_pred_lunch = rf_classifier_lunch.predict(X_test_lunch)

accuracy = accuracy_score(y_test_lunch, y_pred_lunch)
print(f"Accuracy: {accuracy}")

    #using new data point and predicting what lunch is
new_data_lunch = pd.DataFrame({
    'math_score': [100],
    'reading_score':[100],
    'writing_score':[100],
    'averageScore':[100],
    'gender_male': [0],
    'race_ethnicity_group B': [1],
    'race_ethnicity_group C': [0],
    'race_ethnicity_group D': [0],
    'race_ethnicity_group E': [0],
    "parental_level_of_education_bachelor's degree": [0],
    'parental_level_of_education_high school': [1],
    "parental_level_of_education_master's degree": [0],
    'parental_level_of_education_some college': [0],
    'parental_level_of_education_some high school': [0],
    'test_preparation_course_none': [0]
})

new_data_encoded_lunch = pd.get_dummies(new_data_lunch, drop_first=True)

predicted_lunch_type = rf_classifier_lunch.predict(new_data_encoded_lunch)
if(predicted_lunch_type == True):
    predicted_lunch_type = 'Standard'
else:
    predicted_lunch_type = 'Free/Reduced'
print(f"Lunch type: {predicted_lunch_type}")


    #plotting confusion matrix

cm_lunch = confusion_matrix(y_test_lunch, y_pred_lunch)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lunch, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Random Forest Classification)for Lunch')
plt.show()
    #2nd plot feature importance
importances = rf_classifier_lunch.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 8))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), X_encoded_lunch.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance Plot (Random Forest)for Lunch')
plt.show()


#predicting test preperation
X_preparation = df.drop('test_preparation_course', axis=1) 
y_preparation = df['test_preparation_course']

X_encoded_preparation = pd.get_dummies(X_preparation, drop_first=True)
y_encoded_preparation = pd.get_dummies(y_preparation, drop_first=True)
X_train_preparation, X_test_preparation, y_train_preparation, y_test_preparation = train_test_split(X_encoded_preparation, y_encoded_preparation, test_size=0.1, random_state=40)
rf_classifier_preparation = RandomForestClassifier(n_estimators=100, random_state=40)
rf_classifier_preparation.fit(X_train_preparation, y_train_preparation)
y_pred_preparation = rf_classifier_preparation.predict(X_test_preparation)

accuracy = accuracy_score(y_test_preparation, y_pred_preparation)
print(f"Accuracy: {accuracy}")

#using new data point and predicting what preparation is
new_data_preparation = pd.DataFrame({
    'math_score': [100],
    'reading_score':[100],
    'writing_score':[100],
    'averageScore':[100],
    'gender_male': [0],
    'race_ethnicity_group B': [1],
    'race_ethnicity_group C': [0],
    'race_ethnicity_group D': [0],
    'race_ethnicity_group E': [0],
    "parental_level_of_education_bachelor's degree": [0],
    'parental_level_of_education_high school': [1],
    "parental_level_of_education_master's degree": [0],
    'parental_level_of_education_some college': [0],
    'parental_level_of_education_some high school': [0],
    'lunch_standard': [0]
})

new_data_encoded_preparation = pd.get_dummies(new_data_preparation, drop_first=True)

predicted_preparation_type = rf_classifier_preparation.predict(new_data_encoded_preparation)
if(predicted_preparation_type == True):
    predicted_preparation_type = 'not completed'
else:
    predicted_preparation_type = 'completed'
print(f"Preparation type: {predicted_preparation_type}")


    #plotting confusion matrix
cm_preparation = confusion_matrix(y_test_preparation, y_pred_preparation)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_preparation, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Random Forest Classification)for Preparation')
plt.show()
    #2nd plot feature importance
importances = rf_classifier_preparation.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 8))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), X_encoded_preparation.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance Plot (Random Forest)for Preparation')
plt.show()