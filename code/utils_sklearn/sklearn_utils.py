from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from mee import *
import pandas as pd

class CustomGridSearch_cv:
    def __init__(self, estimator, hyperparameters, cv_strategy='stratified', cv_splits=5, multi_output = False, random_state=0):
        self.estimator = estimator
        self.hyperparameters = hyperparameters
        self.cv_strategy = cv_strategy
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.multi_output = multi_output

    def fit(self, X_train, y_train):
        if self.estimator == 'knn':
            base_estimator = KNeighborsClassifier()
        elif self.estimator == 'knn_reg':  
            base_estimator = KNeighborsRegressor()  
        elif self.estimator == 'svc':
            base_estimator = SVC()
        elif self.estimator == 'multi_svr': 
            base_estimator = SVR()
            if self.multi_output:
                base_estimator = MultiOutputRegressor(base_estimator)
        else:
            raise ValueError("Invalid estimator type. Use 'knn', 'knn_reg', 'svc' or 'multi_svr'.")

        if self.cv_strategy == 'stratified':
            cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        elif self.cv_strategy == 'kfold':
            cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        else:
            raise ValueError("Invalid cross-validation strategy. Use 'stratified' or 'kfold'.")

        grid = GridSearchCV(
            base_estimator,
            param_grid=self.hyperparameters,
            cv=cv,
            scoring='accuracy' if self.estimator == 'knn' or self.estimator == 'svc' else make_scorer(MEE, greater_is_better=False),
            verbose=1,
            n_jobs=1,
            return_train_score = True
        )
        grid.fit(X_train, y_train)
        self.best_params_ = grid.best_params_
        self.best_estimator = grid.best_estimator_  
        cv_results = pd.DataFrame(grid.cv_results_)
        best_index = grid.best_index_ 
        train_loss = cv_results['mean_train_score'][best_index]
        train_std = cv_results['std_train_score'][best_index]
        validation_loss = cv_results['mean_test_score'][best_index]
        validation_std = cv_results['std_test_score'][best_index]
        print("Best parameters found:", self.best_params_)

   
    def evaluate(self, X, y):
        if self.best_estimator is None:
            raise ValueError("Fit the model before evaluation.")

        if self.estimator == 'knn' or self.estimator == 'svc':
            accuracy = accuracy_score(y, self.best_estimator.predict(X))
            mse = mean_squared_error(y, self.best_estimator.predict(X))
            return accuracy, mse
        elif self.estimator == 'knn_reg':
            mee = MEE(y, self.best_estimator.predict(X))
            mse = mean_squared_error(y, self.best_estimator.predict(X))
            return mee, mse
    

    