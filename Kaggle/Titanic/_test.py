from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import lib.functions as f
import glob, os
import config


df_train = pd.read_csv(config.values['transformed_data_path'] + config.values['train_file'])
df_test = pd.read_csv(config.values['transformed_data_path'] + config.values['test_file'])

print(df_test.describe())