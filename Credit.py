import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pprint as ppprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#import the dataset
data=pd.read_csv('/Users/cecilia/Desktop/pythonTPs/crx.data', delimiter=',', na_values=['?'])
#check the dataset
with pd.option_context('display.max_rows', None, 'display.max_columns', None): display(data)
#replacing the NAN with the most representitive category or with the average in the case of quatitatve var. 
data['A1']= data['A1'].fillna('b')
data['A2']= data['A2'].fillna(31.56)
data['A4']= data['A4'].fillna('u')
data['A5']= data['A5'].fillna('g')
data['A6']= data['A6'].fillna('c')
data['A7']= data['A7'].fillna('v')
data['A14']= data['A14'].fillna(184.0)
#preprocessing the variables to change the letters by number values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in data.columns:
        if data[column_name].dtype == object:
            data[column_name] = le.fit_transform(data[column_name].astype(str))
        else:
            pass

#testing whether the data was succesfully changed
data['A4'].value_counts()
#plot the histograms for contimuous variables
sns.set()
fig, (ax1, ax2, ax3)= plt.subplots(1, 3)
fig.suptitle('Histograms of continous variables')
ax1.hist(data['A2'])
ax1.set_title('A2')
ax2.hist(data['A3'])
ax2.set_title('A3')
ax3.hist(data['A8'])
ax3.set_title('A8')
plt.show()

sns.set()
fig, (ax1, ax2)= plt.subplots(1, 2)
fig.suptitle('Histograms of continous variables')
ax1.hist(data['A14'])
ax1.set_title('A14')
ax2.hist(data['A15'])
ax2.set_title('A15')
plt.show()

#plot the correlation matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(method='pearson'), annot=True, fmt='.2f', 
            cmap=plt.get_cmap('coolwarm'), cbar=True, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
plt.savefig('result.png')

#separating the features from the target
datay= data[['A16']]
datax= data.drop(labels='A16', axis=1)
#split the data in train and test
X_train, X_test, y_train, y_test = train_test_split(datax, datay, test_size=0.20, random_state=0)

###DECISION TREE CLASSIFIER
#############################
#Tuning the parameters for decision tree (grid search and cross validation)
maxD = [2, 3, 4, 5, 6, 7, 8, 10]
minS = [2, 5, 7, 10, 14, 18, 20, 22]

grid_search = GridSearchCV(tree.DecisionTreeClassifier(), dict(max_depth=maxD,
                 min_samples_split=minS), cv=20)
grid_search.fit(X_train, y_train)
grid_search.best_estimator_.score(X_test, y_test)
#obtained the best parameters
grid_search.best_params_
#plot the resulting tuning
def plot_grid_search(cv_results, maxD, minS, Max_depth, Min_sample):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(minS),len(maxD))
    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(minS),len(maxD))
    _, ax = plt.subplots(1,1)
    for idx, val in enumerate(minS):
        ax.plot(maxD, scores_mean[idx,:], '-o', label= Min_sample + ': ' + str(val))
        
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(Max_depth, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)

plot_grid_search(grid_search.cv_results_, minS,maxD, 'Min sample Split', 'Max Depth')

tree_model = tree.DecisionTreeClassifier(max_depth=4, min_samples_split= 20)
tree_model.fit(X_train, y_train.values.ravel())

#evaluating the models
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    #mape = 100 * np.mean(errors / test_labels)
    #accuracy = 100 - mape
    accuracy = model.score(test_features,test_labels)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}'.format(accuracy))
    
    return accuracy
#obtainig the model performance
best_grid2 = grid_search.best_estimator_
tree_accuracy = evaluate(best_grid2, X_test, y_test.values.ravel())
#obtained the actual classification tree
with open("First.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, out_file=f)
#features importance in decision tree
importances = list(tree_model.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(datax.columns, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'g', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values, datax.columns, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
#############################
###RANDOM FOREST CLASSIFIER
#############################
#fitting the first basic model
base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train.values.ravel())
#evaluating the models
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    #mape = 100 * np.mean(errors / test_labels)
    #accuracy = 100 - mape
    accuracy = model.score(test_features,test_labels)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}'.format(accuracy))
    
    return accuracy
#evaluating the base model
base_accuracy = evaluate(base_model, X_test, y_test.values.ravel())
#TSetting the parameters for tuning
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

#Model tuning
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train.values.ravel())
#obtain the best parameters
rf_random.best_params_
#evaluating the random search model with best estimators
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test.values.ravel())
# print the improvement with respect to base model
print('Improvement of {:0.2f}.'.format( (random_accuracy - base_accuracy) / base_accuracy))
#grid search tuning based on the results of random search
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [40, 50, 70, 100],
    'max_features': [3, 4, 5],
    'min_samples_leaf': [2, 3, 4],
    'min_samples_split': [3, 5, 7],
    'n_estimators': [100, 300, 500, 1600]
}
#fitting the model
rf1 = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf1, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train.values.ravel())
#obtain the best parameters
grid_search.best_params_
#evaluate the performance of grid search model
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test.values.ravel())
# print the improvement with respect to base model
print('Improvement of {:0.2f}'.format((grid_accuracy - base_accuracy) / base_accuracy))
#testing individualy the importance of hyperparameters: maximum features
N = 15
accuracy = np.zeros(N)
for i in range(N):
    model2 = RandomForestClassifier(bootstrap =True,max_depth=70, max_features=(i+1), min_samples_split= 3, min_samples_leaf = 2, n_estimators = 500)
    model2 = model2.fit(X_train, y_train.values.ravel())
    Z = model2.predict(X_test)
    accuracy[i] = model2.score(X_test, y_test.values.ravel())

#plot the results
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
plt.plot([(i+1) for i in range(15)], accuracy,'-g')
ax.set_title("Accuracy vs Max Features", fontsize=20, fontweight='bold')
ax.set_xlabel('Number of features', fontsize=16)
ax.set_ylabel('Accuracy', fontsize=16)
plt.show()

#testing the n_sample in global accuracy of model: Number of estimators
N = 50
accuracy = np.zeros(N)
for i in range(N):
    model2 = RandomForestClassifier(bootstrap =True,max_depth=70, max_features= 3, min_samples_split= 3, min_samples_leaf = 2, n_estimators = (i+1)*10)
    model2 = model2.fit(X_train, y_train.values.ravel())
    Z = model2.predict(X_test)
    accuracy[i] = model2.score(X_test, y_test.values.ravel())

plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
plt.plot([10*(i+1) for i in range(50)], accuracy,'-g')
#plt.plot(c, error, '-g')
#plt.plot(c, b, '-g')
ax.set_title("Accuracy vs Number of estimators", fontsize=20, fontweight='bold')
ax.set_xlabel('Number of estimators', fontsize=16)
ax.set_ylabel('Accuracy', fontsize=16)
plt.show()

#features importance in Random Forest
model3 = RandomForestClassifier(bootstrap =True,max_depth=50, max_features=15, min_samples_split= 3, min_samples_leaf = 2, n_estimators = 300)
model3 = model2.fit(X_train, y_train.values.ravel())
importances = list(model3.feature_importances_)
feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(datax.columns, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values, datax.columns, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

#############################
###EXTREME RANDOMIZED TREE CLASSIFIER (ADABOOSTING)
#############################
#seting the parameters for tuning
p_grid = {
    'learning_rate': [0.8, 1.0, 2.0, 4.0],
    'n_estimators': [100, 300, 500, 1000]
}

#fitting the tuned parameters
ada = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=4, min_samples_split= 20))
Grid_search = GridSearchCV(estimator = ada, param_grid = p_grid, cv = 10, n_jobs = -1, verbose = 2)
Grid_search.fit(X_train, y_train.values.ravel())
#obtain the best parameters
Grid_search.best_params_
#evaluate the performance of the model
best_grid1 = Grid_search.best_estimator_
grid_accuracy1 = evaluate(best_grid1, X_test, y_test.values.ravel())
#print the improvement 
print('Improvement of {:0.2f}'.format((grid_accuracy1 - tree_accuracy) / tree_accuracy))
#features importance in adaboost
model4 = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=4), learning_rate= 1.0, n_estimators = 500)
model4 = model4.fit(X_train, y_train.values.ravel())
importances = list(model4.feature_importances_)
feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(datax.columns, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values, datax.columns, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

