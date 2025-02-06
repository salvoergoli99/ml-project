import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import the MEE function
def mean_euclidean_error(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)))

# Class to train, evaluate, print the curves and save of the model
class Trainer_cup:
    def __init__(self, model, x_train, y_train, x_test=None, y_test=None, target='mee', use_early_stopping=False, early_stopping_patience=10):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.target = target
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = early_stopping_patience
        

        self.histories = []
        self.best_epoch = None
        self.best_model_stats = {}

    def lr_decay(self, lr_decay=0.05, epochs_drop=10, learning_zero=0.005):
        return keras.callbacks.LearningRateScheduler(lambda e: lr_decay * 0.5 ** (e / epochs_drop) + learning_zero)

    def train(self, epochs, batch_size=32, lr_decay=None, epochs_drop=None, learning_zero=None, verbose=None):
        callbacks = []
        """ Trains the model for a specified number of epochs.
        Args:
            epochs (int): The number of epochs to train the model for.
            batch_size (int, optional): The batch size to use for training. Defaults to 32.
            lr_decay (float, optional): The learning rate decay factor. If provided, a LearningRateScheduler callback will be added to the callbacks list. Defaults to None.
            epochs_drop (int, optional): The number of epochs to drop the learning rate by half. If provided, a LearningRateScheduler callback will be added to the callbacks list. Defaults to None.
            learning_zero (float, optional): The minimum learning rate. If provided, a LearningRateScheduler callback will be added to the callbacks list. Defaults to None.
            verbose (int, optional): The verbosity mode. If provided, the verbosity level of the model's fit method will be set to the provided value. Defaults to None.
        Notes:
            - If lr_decay is provided, epochs_drop and learning_zero must also be provided.
            - If early stopping is enabled (self.use_early_stopping is True), an EarlyStopping callback will be added to the callbacks list.
            - If x_test is provided, the model will be trained with validation data.
            - If x_test is not provided, the model will be trained without validation data.
            - The training history is stored in the histories attribute.
            - If the histories attribute is not empty, the loss and metric values at the best epoch are stored in the best_model_stats attribute.
"""
        if lr_decay is not None:
            callbacks.append(self.lr_decay(lr_decay, epochs_drop, learning_zero))

        # Check if early stopping is to be used
        if self.use_early_stopping:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_mean_euclidean_error', 
                patience=self.early_stopping_patience, 
                verbose=verbose, 
                restore_best_weights=True,
                mode='min'
            )
            callbacks.append(early_stopping)

        if self.x_test is not None:
            history = self.model.fit(self.x_train, self.y_train,
                                     epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose,
                                     validation_data=(self.x_test, self.y_test))
        else:
            history = self.model.fit(self.x_train, self.y_train,
                                     epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)

        self.histories.append(history)

        if len(self.histories) > 0:
            last_history = self.histories[-1]

            if self.x_test is not None:
                if self.target == 'loss':
                    best_epoch = np.argmin(last_history.history['val_loss'])
                else:
                    best_epoch = np.argmin(last_history.history['val_mean_euclidean_error'])

                self.best_model_stats['loss'] = last_history.history['val_loss'][best_epoch]
                self.best_model_stats['metric'] = last_history.history['val_mean_euclidean_error'][best_epoch]
                self.best_epoch = best_epoch
        
    def evaluate(self, x_test, y_test):
        if self.model is None:
            raise ValueError('No trained model found. Train the model first.')

        test_loss, test_mee = self.model.evaluate(x_test, y_test, verbose=0)
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Mean Euclidean Error: {test_mee:.4f}')

        return test_loss, test_mee

    def predict(self, x):
        if self.model is None:
            raise ValueError('No trained model found. Train the model first.')

        return self.model.predict(x)

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError('No trained model found. Train the model first.')

        self.model.save(filepath)

        print(f"Model saved to {filepath}")

    def plot_history(self,title_suffix=''):
        if len(self.histories) == 0:
            raise ValueError('No training histories')

        last_history = self.histories[-1]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(last_history.history['loss'], label='Train Loss', linewidth=2, linestyle=':')
        if self.x_test is not None:
            plt.plot(last_history.history['val_loss'], label='Test Loss', linewidth=2)
        plt.title(f'Loss - {title_suffix}', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=10)

        plt.subplot(1, 2, 2)
        plt.plot(last_history.history['mean_euclidean_error'], label='Train MEE', linewidth=2, linestyle=':')
        if self.x_test is not None:
            plt.plot(last_history.history['val_mean_euclidean_error'], label='Test MEE', linewidth=2)
        plt.title(f'Mean Euclidean Error - {title_suffix}', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('MEE', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        plt.show()
