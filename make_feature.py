# Dataset is in https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset
# FiFTy: Large-scale File Fragment Type Identification using Convolutional Neural Networks

import pandas as pd
import numpy as np
import os 
import json
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from tqdm import tqdm
import bz2

def load(scenario=1, block_size='4k', subset='train'):
    if block_size not in ['512', '4k']:
        raise ValueError('Invalid block size!')
    if scenario not in range(1, 7):
        raise ValueError('Invalid scenario!')
    if subset not in ['train', 'val', 'test']:
        raise ValueError('Invalid subset!')

    data_dir = os.path.join('./', '{:s}_{:1d}'.format(block_size, scenario))
    data = np.load(os.path.join(data_dir, '{}.npz'.format(subset)))

    if os.path.isfile('./classes.json'):
        with open('./classes.json') as json_file:
            classes = json.load(json_file)
            labels = classes[str(scenario)]
    else:
        raise FileNotFoundError('Please download classes.json to the current directory!')

    return data['x'], data['y'], labels

x, y, labels = load(1, '512', 'train')

train_data_size = 500000
x = x[:train_data_size]
y = y[:train_data_size]
print("Loaded data: x.shape={}, y.shape={}".format(x.shape, y.shape))

print(str(len(labels)), 'labels')

print(x[0])
print(y[0])

# Feature 만드는 부분
print('start making feature')
from scipy.stats import hmean, entropy
from collections import Counter
from scipy.stats import skew, kurtosis

def complexity(x):
    comp = bz2.compress(bytes(x))
    return len(comp)

def ASCII_Range_Freq(x):
    a = b = c = 0
    for i in x:
        if 0 <= i < 32 :
            a += 1
        elif 32 <= i < 128 :
            b += 1
        else:
            c += 1
    return a,b,c

def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())

def hammingWeight(n):
    n = str(bin(n))
    c = Counter(list(n))
    return c['1']

def avg_hamming(xs):
    xslen = len(xs)
    total = 0
    for x1 in xs:
        total += hammingWeight(x1)
    return total/xslen

def longest_streak(x):
    prev = x[0]
    streak = 1
    max_s = 1
    for i in range(1, len(x)):
        if x[i] != prev:
            streak = 1
        else:
            streak += 1
            if max_s < streak:
                max_s = streak
        prev = x[i]
    return max_s

def get_mean_absolute_deviation(x):
    return np.mean(np.absolute(x - np.mean(x, axis=0)), axis=0)

def get_entropy(x):
    pd_series = pd.Series(x)
    counts= pd_series.value_counts()
    return entropy(counts)


# 한개의 데이터에 대해 preprocessing 후 배열 반환(순서 유의)
def preprocessing(x):
    no_zero_place = np.where(x==0)
    no_zero_x = np.array(x)
    no_zero_x[no_zero_place] = 1

    #feature 만들고
    comp = complexity(x)
    arith_mean = np.mean(x)
    geo_mean = geo_mean_overflow(no_zero_x)
    harm_mean = hmean(no_zero_x)
    std_dev = np.std(x)
    mad = get_mean_absolute_deviation(x)
    hamm_w = avg_hamming(x)
    kurto = kurtosis(x)
    skw = skew(x)
    l_strk = longest_streak(x)
    LARF, MARF, HARF = ASCII_Range_Freq(x)
    ent = get_entropy(x)    
    #feature 순서대로 넘기기(논문 순서)
    return [comp, arith_mean, geo_mean, harm_mean, std_dev, mad, hamm_w, kurto, skw, l_strk, LARF, MARF, HARF, ent]

# 데이터셋에 대해 Feature 데이터셋 생성 후 반환
def make_feature_dataset(xs):
    feature_dataset = []
    for x1 in tqdm(xs):
        feature_dataset.append(preprocessing(x1))
    return feature_dataset
  
feature_data = make_feature_dataset(x)


train_df = DataFrame(feature_data, columns = ["Kolmogrov_Complexity", "Arithmetic_Mean", "Geometric_Mean", "Harmonic_Mean", 
                                              "Standard_Deviation", "Mean_Absolute_Deviation", 
                                              "Hamming_Weight", "Kurtosis", "Skewness", 
                                              "Longest_Byte_Streak", "Low_ASCII_Range_Freq" , 
                                              "Med_ASCII_Range_Freq" ,"High_ASCII_Range_Freq", "Shannon_Entropy"])
print('train_df complete')
print(train_df)

train_df.to_csv('./feature_data/train_x.csv', index=False)
np.save('./feature_data/train_y.npy', y)

#Test phase------------------------------
print('test data load')
test_data_size = 100000

x_test, y_test, labels = load(1, '512', 'test')
x_test = x_test[:test_data_size]
y_test = y_test[:test_data_size]
print("Loaded data: x.shape={}, y.shape={}".format(x_test.shape, y_test.shape))

test_feature_data = make_feature_dataset(x_test)
test_df = DataFrame(test_feature_data, columns = ["Kolmogrov_Complexity", "Arithmetic_Mean", "Geometric_Mean", "Harmonic_Mean", 
                                              "Standard_Deviation", "Mean_Absolute_Deviation", 
                                              "Hamming_Weight", "Kurtosis", "Skewness", 
                                              "Longest_Byte_Streak", "Low_ASCII_Range_Freq" , 
                                              "Med_ASCII_Range_Freq" ,"High_ASCII_Range_Freq", "Shannon_Entropy"])
print(test_df)
print('test_df complete')

test_df.to_csv('./feature_data/test_x.csv', index=False)
np.save('./feature_data/test_y.npy', y_test)