# ComparingClassifiers
## Problem Statement:
Compare the performance of 4 classifiers KNN, Logistic Regression, Decision Tree and Support Vector Machine using a dataset related to marketing bank products over phone.
### Analysis
The dataset has 41188 records with 21 features including numerical and non-numerical columns as described below:
Input variables:
##### bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')

##### related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

##### other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

##### social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

##### Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

#### Data Cleaning
From the description of 'Duration' column, as it highly affects the output target, 'duration' column is removed.
I checked if any column had null values and none of the columns contain null value.
I checked the unique values for the non-numeric columns and from the unique values, they seem to be categorial columns.

### EDA
1. Prepare the datasets:
   a) First, I split the dataset into 2 parts: i) X contains input features and ii) y contains output variable.
   b) Encoded and transformed target variable from 'yes/no' to '1/0'.
   c) Identified categorial and numerical columns in separate lists.
   d) Initiated column transformers using StandardScaler() for numerical columns and OneHotEndcoder for categorial columns
   e) Split the dataset into train and test datasets using train_test_split with 80% data into train and 20% into test dataset.
   
2. Baseline Model for baseline comparison:
   I have used DummyClassifier to create a baseline score to compare with other 4 classifiers' score. The baseline score is 0.8865 so around 89%.
   
4. Compare 4 classifiers performance
   i) I started with Logistic Regression and the test score is 0.8969. The result is then added to results[].
   ii) I have created 'models' dictionary to hold model name and constructors with default arguments and then created for look that goes through each model and append the fit time, training score and         test score for each model to results[].
   
### Results
Result1:
The results list is converted to json for better visualization of table as below.
| Model | Train Time (s) | Train Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.207447 | 0.901366 | 0.896941 |
| KNN | 0.089821 | 0.914264 | 0.891964 |
| SVC | 60.491481 | 0.905432 | 0.897062 |
| Decision Tree Classifier | 0.513193 | 0.995357 | 0.836004 |

### Improvement
#### Hyperparameters tuning and GridSearchCV
1) As first part of improvement, I tried running all 4 models with default parameters (except LogisticRegression with max iterations = 1000) along with GridSearchCV and results are as below.
Result2:

| Model | Train Time | Train Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| knn | 10.541481 | 0.914264 | 0.891964 |
| logisticregression | 2.361602 | 0.901366 | 0.896941 |
| svc | 104.154096 | 0.905432 | 0.897062 |
| decisiontreeclassifier | 3.176454 | 0.995357 | 0.838432 |

Observation: Comparing Result1 and Result2, it looks Train time increased with GridSearchCV, specially for knn and svc, however Test Accuracy stayed around the same for all models.

2) To improve the test score and train time, I tried using GridSearchCV with tuning the models with different hyperparameters (knn__n_neighbors =7 -> default is 5, logisticregression__c: [0.1,10] -> default is 1, decisiontreeclassifier__max_length: [5,10] -> default was no max length restriction) and obtained the results as below:
Result3:

| Model | Train Time (s) | Train Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **KNN** | 7.112850 | 0.909712 | 0.892207 |
| **Logistic Regression** | 0.300718 | 0.901396 | 0.897427 |
| **SVC** | 1588.573161 | 0.898209 | 0.894756 |
| **Decision Tree Classifier** | 2.810019 | 0.904492 | 0.896213 |

Observation: The train time reduced for knn, logistic regression and decision tree classifiers while their test score remained almost same - it has very small (0.1%) improvement for knn and logistic regression) and it increased by 6% for decision tree classifier.
The train time drastically increased for svc even though accuracy decreased by around 0.3%.

3) As I see the Test Accuracy increasing with knn and decision tree, I tried different parameters for knn and decision tree as their train time are much less than svc as well.
Updating knn__n_neighbors to 3.
Result4:

| Model | Train Time (s) | Train Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: |
| **knn** | 7.112850 | 0.909712 | 0.892207 |
| **logisticregression** | 0.300718 | 0.901396 | 0.897427 |
| **svc** | 1588.573161 | 0.898209 | 0.894756 |
| **decisiontreeclassifier** | 2.810019 | 0.904492 | 0.896213 |
| **knnWithNeighbor3** | 11.312925 | 0.924825 | 0.881160 |

Observation: the train time is increased for knn and test accuracy is decreased.

4) Decision Tree hyperparameters: I tried max_depth = 10, min_samples_leaf = 1 and min_samples_split = 2. The train time improved very well but the test score went down by 0.6%.
Result5:

| Model | Train Time | Train Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: |
| **knn** | 7.112850 | 0.909712 | 0.892207 |
| **logisticregression** | 0.300718 | 0.901396 | 0.897427 |
| **svc** | 1588.573161 | 0.898209 | 0.894756 |
| **decisiontreeclassifier** | 2.810019 | 0.904492 | 0.896213 |
| **knnWithNeighbor3** | 11.312925 | 0.924825 | 0.881160 |
| **decisiontree2** | 2.573698 | 0.917147 | 0.890265 |

Observation: the train time is decreased a tiny bit for decision tree and test accuracy is increased by 0.6%.

#### Re analyzing features
knn and svc have higher train time and test score is not improving much for any model so I analyzed feature correlation to analyze feature selection. 
5) For categorial features, I plotted distribution of each feature by target variable count using seaborn histplot. By looking at the target variable count distribution, it looks like each feature is distributed for each target value 0 and 1. 'day of week' column has very similar distribution across target variable y but it's not exactly same.
   
6) For numerical features, I plotted correlation matrix heatmap to see their correlation values. From the plot, I see that cons.conf.idx(score =0.05), age(0.03) and campaign(0.07) have very low correlation score with target variable 'y'. However, this is linear correlation value so there is a possibility that combining them with other features might have different impact on target variable so not removing them right now. None of the features shows strong relationship with target in this plot.
   
7) As I didn't see strong correlation that can be removed, I tried different direction: encoding categorial features manually to numerical values using 'replace' on the dataframe columns - job, marital, education, day_of_week and month. I want to see if having all numerical values improve training time and test score.
I ran correlation matrix on this new dataframe with all numerical columns and dropped the columns ('euribor3m', 'nr.employed') that had score either >0.85 or <-0.85 to remove strongly correlated features from this matrix.

8) I repeated the process again:
   a) baseline model to compare scores with. Test score is now 0.8865. This is same as previous baseline score.
   b) Train all 4 classifiers with default parameters and without GridSearchCV:
   Result6:

| Model | Train Time (s) | Train Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: |
| **logisticregression** | 1.914592 | 0.907284 | 0.907016 |
| **knn** | 0.003197 | 0.925645 | 0.899733 |
| **svc** | 7.855997 | 0.902640 | 0.901554 |
| **decisiontreeclassifier** | 0.097564 | 1.000000 | 0.879461 |

Observation: Compare to Result1 (using OneHotEncode method), Result6 (manually converting categories from text to numbers within same column) show improvement in both time for most and test score for all.

	c) Train all 4 classifiers with different hyperparameters:
	Result7:

| Model | Train Time (s) | Train Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: |
| **KNN** | 3.184482 | 0.920759 | 0.901190 |
| **Logistic Regression** | 2.394371 | 0.907436 | 0.906895 |
| **SVC** | 159.389934 | 0.905038 | 0.899733 |
| **Decision Tree Classifier** | 0.957584 | 0.912868 | 0.911872 |

	Observation: Compare to Result3, the Test accuracy is improved. Compare to Result6, knn, svc and Decision tree have better Test Accuracy.
	
   d) As the decision tree classifier is showing the most improvement in Test accuracy and Train time is very less, I tried decision tree classifier with different hyperparameters: max_depth: [5,10,15], min_samples_lead : [1,2,4], min_samples_split: [1,2].
	Result8:
| Model | Train Time | Train Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| knn | 3.184482 | 0.920759 | 0.901190 |
| logisticregression | 2.394371 | 0.907436 | 0.906895 |
| svc | 159.389934 | 0.905038 | 0.899733 |
| decisiontreeclassifier | 0.957584 | 0.912868 | 0.911872 |
| decisiontree-bestfeatures | 0.279350 | 0.912868 | 0.911872 |

	Observation: Test Accuracy stayed the same for Decision tree.

#### findings
The business problem is to predict whether the customer will subscribe to the campaign given the features in the dataset. This is classification problem and the intention is to compare the performance of 4 classifiers - k-nearest neighbors, logistic regression, decision trees, and support vector machines.

From the analysis, the test score is improved after converting all categorial features to numerical features. This method reduces the number of columns and hence train time is also reduced. While comparing 4 models, if I used OneHotEncoder to convert categories into separate columns,  Logistic Regression had best test score of 89.74% and train time of 0.300718 seconds.
If I converted all categorial columns to numerical columns, Decision Tree had the best train time of 0.279350 seconds and best test score of 91.18%.

It's clear that models performed better with less number of columns - after converting categorial columns to numerical columns by replacing their text value to numerical representation in original column. With this approach, among all 4 classifiers, Decision Tree is showing best Test Accuracy and with least train time.

At last, from the evaluation matrix of the last Decision Tree model implemented here, Predicted Negative is 7015 out of actual Negative of (7015+288) 7303. Predicted Positive is 497 out of (438 + 497) 935. This means that this model predicts the negative value for target column 'y' very well, however, it's poor at predicting positive values out of total positive. It predicted around 53% positive out of total positive target values.

##### Recommendations
If I were to continue this analysis, I would further explore decision tree classifier by analyzing feature_importances and removing some features to see if Test Accuracy improves. However, to keep in mind, features that can have lower importance for decision tree might have different weightage for different classifier.
I would also explore more hyperparameter tuning for knn and logistic regression as their test score have improved as well.

Link to Jupiter Notebook: https://github.com/ShivaniD719/ComparingClassifiers/blob/main/CompareClassifiers.ipynb
