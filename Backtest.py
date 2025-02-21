# Import Required Libraries
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from pykalman import KalmanFilter
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")
