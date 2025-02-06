import pandas as pd
from sklearn.model_selection import train_test_split

cols_Cup = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 't1', 't2', 't3']



class CupDataset:

  def __init__(self,name, data_dir='/Users/silviocalderarao/Downloads/Monks/Dataset/'):
    self.name = name
    self.data_dir = data_dir

    if name == 'Cup_tr':
        self.data = pd.read_csv(f'{self.data_dir}/ML-CUP23-TR.csv', comment="#",  names= cols_Cup) #header= None, names=cols_Cup, sep='')
    elif name == 'Cup_ts':
        self.data = pd.read_csv(f'{self.data_dir}/ML-CUP23-TS.csv', comment='#', names= cols_Cup)
    else:
      raise ValueError('Invalid dataset name')

    self.X = self.data.drop(['id','t1','t2', 't3'], axis=1)
    self.y = self.data. drop(['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10'], axis=1)

  def split_data(self, test_size=0.1, random_state=None):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=test_size, random_state=random_state)

  def get_splits(self):
    X_train = pd.DataFrame(self.X_train, columns=self.X.columns)
    X_test = pd.DataFrame(self.X_test, columns=self.X.columns)
    y_train = pd.DataFrame(self.y_train, columns=['t1', 't2', 't3'])
    y_test = pd.DataFrame(self.y_test, columns=['t1', 't2', 't3'])

    return X_train, X_test, y_train, y_test

def get_cup_data(train_set, test_set):
  X_dev = train_set.X
  y_dev = train_set.y

  X_test = test_set.X
  y_test = test_set.y

  return X_dev, y_dev, X_test, y_test

