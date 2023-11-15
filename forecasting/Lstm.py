import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from math import sqrt
from scipy.io import loadmat
import scipy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import random
import copy
import math


from transformer.decoder import Decoder
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.pointerwise_feedforward import PointerwiseFeedforward
from transformer.encoder_decoder import EncoderDecoder
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
from transformer.batch import subsequent_mask
from transformer.noam_opt import NoamOpt

# check cuda is available
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def CreateSubwindows(df, n=100):
  """Creates running windows to interpolate the DataFrame.
  It takes 1 point every n to low the resolution.

  Args:
  ----------
  df : DataFrame
        Input DataFrame
  n : int (default=100)
        Number of points to take in lowering the resolution.

  Returns:
  ----------
  df : resulting DataFrame
  """

  # running windows to interpolate the df
  df=df.rolling(1000).apply(lambda w: scipy.stats.trim_mean(w, 0.05)) #mean without outliers (5 and 95 percentile)
  df=df[1000:(len(df))]
  df=df.reset_index()
  df=df.drop(['index'], axis=1)
  # now we take 1 point every n to low the resolution
  subwindows=[list(i) for i in zip(*[df.values.reshape(-1)[i:i+n] for i in range(0, len(df.values.reshape(-1)), n)])]
  df = pd.DataFrame(subwindows)
  df=df.T
  df=df.apply(np.float32)
  print("A plot of example:")
  plt.plot(df[40])
  plt.show()
  return df
