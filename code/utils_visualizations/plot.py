import  matplotlib.pyplot as plt
def plot_history_cup(train_loss, test_loss, train_mee, test_mee=None, title_suffix=''):

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss', linewidth=2, linestyle=':')

    plt.title(f'Loss - {title_suffix}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=10)

    plt.subplot(1, 2, 2)
    plt.plot(train_mee, label='Train MEE', linewidth=2, linestyle=':')

    plt.title(f'Mean Euclidean Error - {title_suffix}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MEE', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.show()

# This method plots the training history of a neural network model.
# Parameters:
#   - name: str, the name of the dataframe.
#   - train_loss: list, the training loss values.
#   - test_loss: list, the test loss values (optional).
#   - train_accuracy: list, the training accuracy values.
#   - test_accuracy: list, the test accuracy values (optional).

def plot_history_monk(name, train_accuracy, test_accuracy, train_loss, test_loss):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss', linewidth=2, linestyle=':')
    plt.plot(test_loss, label='Test Loss', linewidth=2)
    plt.legend(fontsize=12)
    plt.title(f'Loss - {name}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=10)

   
    plt.subplot(1, 2, 2)

    plt.plot(train_accuracy, label='Train Accuracy', linewidth=2, linestyle=':')

    plt.plot(test_accuracy, label='Test Accuracy', linewidth=2)
    plt.legend(fontsize=12)
    plt.title(f'Accuracy - {name}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.show()