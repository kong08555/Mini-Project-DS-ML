# To import libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# To load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# To separate train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# To define algorithm
KNN = KNeighborsClassifier()

# To define hyper-parameters
param_grid = {
    'n_neighbors': np.arange(1,50),
    'weights':['uniform','distance'],
    'metric':['euclidean','manhattan','minkowski'],
    'p':[1, 2, 3]           
}

# To use grid search for hyper-paramter tuning
grid_search = GridSearchCV(KNN, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# To print the best parameters 
print ("Best Parameters: ", grid_search.best_params_)

# To use the best parameters for prediction
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

# To print the accuracy and classification report (model's performance)
print ("Accuracy: ", accuracy_score(y_test, y_pred))
print (classification_report(y_test, y_pred))