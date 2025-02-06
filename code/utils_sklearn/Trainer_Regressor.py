from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import  RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
import numpy as np
import matplotlib.pyplot as plt
from mee import *



class NeuralNetTrainer_cup:
    def __init__(self, model, param_distributions, X_dev, y_dev, X_test, y_test, cv_splits=5, random_state=42):
        self.model = model
        self.param_distributions = param_distributions
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.X_test = X_test
        self.y_test = y_test
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.best_params_ = None
        self.mlp_best = None
        self.train_mee_epochs =[]
        self.train_loss_epochs = []
        self.test_loss_epochs = []
        self.test_mee_epochs = []
    
    def _search_best_params(self, search_method='grid',n_iter = 10):
        cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        scorer = make_scorer(MEE, greater_is_better=False)
        if search_method == 'grid':
            grid_search = GridSearchCV(
                estimator=self.model, 
                param_grid=self.param_distributions, 
                scoring=scorer, 
                n_jobs=-1, 
                cv=cv, 
                verbose=3)
            grid_search.fit(self.X_dev, self.y_dev)
            self.best_params_ = grid_search.best_params_
            return grid_search.cv_results_
        elif search_method == 'random':
            random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_distributions,
                scoring=scorer,
                n_iter=n_iter,
                n_jobs=-1,
                cv=cv,
                verbose=3,
                random_state=self.random_state
            )
            random_search.fit(self.X_dev, self.y_dev)
            self.best_params_ = random_search.best_params_
            return random_search.cv_results_
        else:
            raise ValueError("Invalid search method. Choose between 'grid' and 'random'.")
    
    def train(self, search_method='grid', epochs=400):
        if not self.best_params_:
            self._search_best_params(search_method)
        
        self.mlp_best = MLPRegressor(random_state=self.random_state, **self.best_params_)
        
        if self.best_params_["early_stopping"]:
          self.mlp_best.fit(self.X_dev, self.y_dev)
        else:
          for epoch in range(epochs):
              self.mlp_best.partial_fit(self.X_dev, self.y_dev)

              y_train_pred = self.mlp_best.predict(self.X_dev)
              train_mee= mean_euclidean_error(self.y_dev, y_train_pred)
              train_loss = mean_squared_error(self.y_dev, y_train_pred)

              self.train_mee_epochs.append(train_mee)
              self.train_loss_epochs.append(train_loss)

              y_test_pred = self.mlp_best.predict(self.X_test)
              test_mee = mean_euclidean_error(self.y_test, y_test_pred)
              test_loss = mean_squared_error(self.y_test, y_test_pred)

              self.test_mee_epochs.append(test_mee)
              self.test_loss_epochs.append(test_loss)

              print(f"Epoch {epoch+1}: Train MEE: {train_mee}, Train Loss: {train_loss}, Test MEE: {test_mee}, Test Loss: {test_loss}")
      
    def plot_training_curves(self):
        plt.figure(figsize=(12,5))
        plt.subplot(121)
        plt.plot(self.train_mee_epochs, label='Training mee')
        plt.plot(self.test_mee_epochs, linestyle='--', label='Validation mee')
        plt.xlabel('Epoch')
        plt.ylabel('mee')
        plt.legend()
        plt.subplot(122)
        plt.plot(self.train_loss_epochs, label='Training loss')
        plt.plot(self.test_loss_epochs, linestyle='--', label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()