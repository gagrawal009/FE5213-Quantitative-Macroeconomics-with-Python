import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd

# AND NOW WE START WORKING WITH REAL DATA
class GoyalWelchData:
  def __init__(self, input_file, symbol):
    self.read_data(input_file, symbol)

  def read_data(self, input_file, symbol):
    

    goyal_welch_data = pd.read_csv(os.path.join('../macro_data','GoyalWelchPredictorData2022Monthly.csv'), index_col=0)
    goyal_welch_data.index = pd.to_datetime(goyal_welch_data.index, format='%Y%m')
    
    monthly_returns = pd.read_csv(os.path.join('../macro_data',input_file))
    monthly_returns.index = pd.to_datetime(monthly_returns.Date, format='%Y-%m')

    for column in goyal_welch_data.columns:
        goyal_welch_data[column] = [float(str(x).replace(',', '')) for x in goyal_welch_data[column]]
    
    self.monthly_returns = monthly_returns
    self.goyal_welch_data = goyal_welch_data
    self.get_cleaned_data(input_file, symbol)
    self.select_signals()

  def get_cleaned_data(self, input_file, symbol):
    start_date = np.max([self.goyal_welch_data.index.min(), self.monthly_returns.index.min()])
    end_date = np.min([self.goyal_welch_data.index.max(), self.monthly_returns.index.max()])
    self.goyal_welch_data = self.goyal_welch_data.loc[start_date:end_date]
    self.goyal_welch_data['excess_returns'] = self.monthly_returns[symbol] - self.goyal_welch_data.Rfree
    self.cleaned_data = self.goyal_welch_data.loc[start_date:end_date].drop(columns=['csp']).fillna(0)

  def normalize(self,
                data: np.ndarray,
                ready_normalization: dict = None,
                use_std: bool = False):
    """

    """

    if ready_normalization is None:
        data_std = data.std(0)
        if use_std:
          data = data / data_std
        else:
          data_max = np.max(data, axis=0)
          data_min = np.min(data, axis=0)
    else:
        data_std = ready_normalization['std']
        if use_std:
          data = data / data_std
        else:
          data_max = ready_normalization['max']
          data_min = ready_normalization['min']

    data = data - data_min
    data = data/(data_max - data_min)
    data = data - 0.5
    normalization = {'std': data_std,
                      'max': data_max,
                      'min': data_min}
    return data, normalization

  def select_signals(self):
    signal_columns = ['Index', 'D12', 'E12', 'b/m', 'tbl', 'AAA', 'BAA', 'lty', 'ntis',
        'Rfree', 'infl', 'ltr', 'corpr', 'svar']
    cleaned_data = self.cleaned_data
    data_for_signals = cleaned_data[signal_columns].shift(1).fillna(0)
    labels = cleaned_data.excess_returns.values.reshape(-1, 1)
    signals = data_for_signals.values
    self.signals = signals

  def get_train_and_test_data(self, normalize_raw_data=True, cheat_and_use_future_data=False):
    signals = self.signals
    labels = self.cleaned_data.excess_returns.values.reshape(-1, 1)
    split = int(signals.shape[0] / 2)
    train_labels = labels[:split]
    test_labels = labels[split:]
    test_dates = self.cleaned_data.excess_returns.index[split:]

    if normalize_raw_data:
        signals[:split, :], normalization = self.normalize(signals[:split])
        if cheat_and_use_future_data:
          signals[split:, :] = self.normalize(signals[split:, :])[0]
        else:
          signals[split:, :] = self.normalize(signals[split:, :],
                                        ready_normalization=normalization)[0]
    train_data = signals[:split, :]
    test_data = signals[split:, :]
    return train_data, test_data, train_labels, test_labels, test_dates

  def sharpe_ratio(x):
    return np.round(np.sqrt(12) * x.mean(0) / x.std(0), 2)
