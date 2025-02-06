from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetTrainer:
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
        self.train_accuracy_epochs = []
        self.train_loss_epochs = []
        self.test_accuracy_epochs = []
        self.test_loss_epochs = []
    
    def _search_best_params(self, search_method='grid'):
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        
        if search_method == 'grid':
            grid_search = GridSearchCV(
                estimator=self.model, 
                param_grid=self.param_distributions, 
                scoring='accuracy', 
                n_jobs=-1, 
                cv=cv, 
                verbose=1
            )
            grid_search.fit(self.X_dev, self.y_dev)
            self.best_params_ = grid_search.best_params_
            print("Best hyperaparameters found:", self.best_params_)
        elif search_method == 'random':
            random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_distributions,
                scoring='accuracy',
                n_iter=500,
                n_jobs=-1,
                cv=cv,
                verbose=2,
                random_state=self.random_state
            )
            random_search.fit(self.X_dev, self.y_dev)
            self.best_params_ = random_search.best_params_
            print("Best hyperaparameters found:", self.best_params_)
        else:
            raise ValueError("Invalid search method. Choose between 'grid' and 'random'.")
    
    def train(self, search_method='grid', epochs=400):
        if not self.best_params_:
            self._search_best_params(search_method)
        
        self.mlp_best = MLPClassifier(random_state=self.random_state, **self.best_params_)
        
        for epoch in range(epochs):
            self.mlp_best.partial_fit(self.X_dev, self.y_dev, classes=np.unique(self.y_dev))

            y_train_pred = self.mlp_best.predict(self.X_dev)
            train_accuracy = accuracy_score(self.y_dev, y_train_pred)
            train_loss = mean_squared_error(self.y_dev, y_train_pred)

            self.train_accuracy_epochs.append(train_accuracy)
            self.train_loss_epochs.append(train_loss)

            y_test_pred = self.mlp_best.predict(self.X_test)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            test_loss = mean_squared_error(self.y_test, y_test_pred)

            self.test_accuracy_epochs.append(test_accuracy)
            self.test_loss_epochs.append(test_loss)

            
        self.train_accuracy_mean = np.mean(self.train_accuracy_epochs)
        self.train_accuracy_std = np.std(self.train_accuracy_epochs)
        self.train_loss_mean = np.mean(self.train_loss_epochs)
        self.train_loss_std = np.std(self.train_loss_epochs)
        
        self.test_accuracy_mean = np.mean(self.test_accuracy_epochs)
        self.test_accuracy_std = np.std(self.test_accuracy_epochs)
        self.test_loss_mean = np.mean(self.test_loss_epochs)
        self.test_loss_std = np.std(self.test_loss_epochs)

        print(f"Train Accuracy: Mean = {self.train_accuracy_mean}, Std = {self.train_accuracy_std}")
        print(f"Train Loss: Mean = {self.train_loss_mean}, Std = {self.train_loss_std}")
        print(f"Test Accuracy: Mean = {self.test_accuracy_mean}, Std = {self.test_accuracy_std}")
        print(f"Test Loss: Mean = {self.test_loss_mean}, Std = {self.test_loss_std}")    
    def plot_training_curves(self):
        plt.figure(figsize=(12,5))
        plt.subplot(121)
        plt.plot(self.train_accuracy_epochs, label='Training accuracy')
        plt.plot(self.test_accuracy_epochs, linestyle='--', label='Validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(122)
        plt.plot(self.train_loss_epochs, label='Training loss')
        plt.plot(self.test_loss_epochs, linestyle='--', label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
