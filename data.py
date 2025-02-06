import pandas as pd
from sklearn.model_selection import train_test_split

cols = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'data_id']
cols_Cup = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 't1', 't2', 't3']

class MonksDataset:

  def __init__(self, name, data_dir='/Users/silviocalderarao/Downloads/Monks/Dataset'):
    self.name = name
    self.data_dir = data_dir

    if name == 'monk1_train':
      self.data = pd.read_csv(f'{self.data_dir}/monks1.train',header=None, names=cols, sep=' ')
    elif name == 'monk1_test':
      self.data = pd.read_csv(f'{self.data_dir}/monks-1.test',header=None, names=cols, sep=' ')
    elif name =='monk2_train':
      self.data = pd.read_csv(f'{self.data_dir}/monks-2.train',header=None, names=cols, sep=' ')
    elif name =='monk2_test':
      self.data = pd.read_csv(f'{self.data_dir}/monks-2.test',header=None, names=cols, sep=' ')
    elif name =='monk3_train':
      self.data = pd.read_csv(f'{self.data_dir}/monks-3.train',header=None, names=cols, sep=' ')
    elif name =='monk3_test':
      self.data = pd.read_csv(f'{self.data_dir}/monks-3.test',header=None, names=cols, sep=' ')
    else:
      raise ValueError('Invalid dataset name')

    self.data = self.data.reset_index(drop=True)
    self.X = self.data.drop(['class', 'data_id'], axis=1)
    self.y = self.data['class']

  def split_data(self, test_size=0.3, random_state=42):
    """Split loaded data into train and test sets"""
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=test_size, random_state=random_state)
    
  def get_splits(self):
    """Return split data"""
    X_train = pd.DataFrame(self.X_train, columns=self.X.columns)
    X_test = pd.DataFrame(self.X_test, columns=self.X.columns)
    y_train = pd.DataFrame(self.y_train, columns=['class'])
    y_test = pd.DataFrame(self.y_test, columns=['class'])
    
    return X_train, X_test, y_train, y_test

# Assuming 'cols' is already defined with the feature names
def get_monks_data(train_set, test_set):
  X_dev = train_set.X
  y_dev = train_set.y

  X_test = test_set.X 
  y_test = test_set.y
  
  return X_dev, y_dev, X_test, y_test

