# Nuova classe 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import callbacks


class Trainer:

  def __init__(self, model, x_train, y_train, x_test=None, y_test=None,
              target='accuracy', dataframe_name=None):
    self.model = model
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.target = target
    self.dataframe_name = dataframe_name

    self.histories = []
    self.best_epoch = 0
    self.best_model_stats = {}

  def lr_decay(self, lr_decay, epochs_drop, learning_zero):
    return keras.callbacks.LearningRateScheduler(lambda e: lr_decay ** (e / epochs_drop) + learning_zero)

  def train(self, epochs, batch_size=32, lr_decay=None, epochs_drop=None, learning_zero=None):
    callbacks = []
    if lr_decay is not None and epochs_drop is not None and learning_zero is not None:
      callbacks.append(self.lr_decay(lr_decay, epochs_drop, learning_zero))

    if self.x_test is not None:  # Using x_test instead of x_val
      history = self.model.fit(self.x_train, self.y_train,
                               epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                               validation_data=(self.x_test, self.y_test))  # Using validation_data with x_test and y_test
    else:
      history = self.model.fit(self.x_train, self.y_train,
                               epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    self.histories.append(history)

    if len(self.histories) > 0:
      last_history = self.histories[-1]

      if self.x_test is not None:  # Using x_test instead of x_val
        if self.target == 'loss':
          best_epoch = np.argmin(last_history.history['val_loss'])
        else:
          best_epoch = np.argmax(last_history.history['val_accuracy'])

        self.best_model_stats['loss'] = last_history.history['val_loss'][best_epoch]
        self.best_model_stats['metric'] = last_history.history['val_accuracy'][best_epoch]

      self.best_epoch = best_epoch
  
  
  def plot_history(self, dataframe_name):
      if len(self.histories) == 0:
        raise ValueError('No training histories')

      last_history = self.histories[-1]

      plt.figure(figsize=(12, 6))
      plt.subplot(1, 2, 1)
      plt.plot(last_history.history['loss'], label='Train Loss', linewidth=2, linestyle=':')
      if self.x_test is not None:  # Using x_test instead of x_val
        plt.plot(last_history.history['val_loss'], label='Test Loss', linewidth=2)  # Using 'Test Loss'
      plt.legend(fontsize=12)
      plt.title(f'Loss - {dataframe_name}', fontsize=14)
      plt.xlabel('Epochs', fontsize=12)
      plt.ylabel('Loss', fontsize=12)
      plt.grid(True)
      plt.tick_params(axis='both', labelsize=10)

      plt.subplot(1, 2, 2)
      plt.plot(last_history.history['accuracy'], label='Train Accuracy', linewidth=2, linestyle=':')
      if self.x_test is not None:  # Using x_test instead of x_val
        plt.plot(last_history.history['val_accuracy'], label='Test Accuracy', linewidth=2)  # Using 'Test Accuracy'
      plt.legend(fontsize=12)
      plt.title(f'Accuracy - {dataframe_name}', fontsize=14)
      plt.xlabel('Epochs', fontsize=12)
      plt.ylabel('Accuracy', fontsize=12)
      plt.grid(True)
      plt.tick_params(axis='both', labelsize=10)

      plt.tight_layout()
      plt.show()

  def predict(self, x):
    if self.model is None:
      raise ValueError('No trained model found. Train the model first.')
    return self.model.predict(x)

  def evaluate(self, x_test, y_test):
    if self.model is None:
      raise ValueError('No trained model found. Train the model first.')
    test_loss, test_acc = self.model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')