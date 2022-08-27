
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import plot_importance

train_df = pd.read_csv('./feature_data/train_x.csv')
train_y = np.load('./feature_data/train_y.npy')

test_df = pd.read_csv('./feature_data/test_x.csv')
test_y = np.load('./feature_data/test_y.npy')

lr = 0.02
# drop_list = ['Arithmetic_Mean', 'Geometric_Mean', 'High_ASCII_Range_Freq', 'Mean_Absolute_Deviation']
# train_df = train_df.drop(drop_list, axis=1)
# test_df = test_df.drop(drop_list, axis=1)

lgbm = LGBMClassifier(num_leaves = 16, learning_rate = lr, max_depth=6, n_estimators=500)
lgbm.fit(train_df, train_y)


print('predict start')
pred = lgbm.predict(test_df)
print(pred.shape)
print(pred)
print(test_y.shape)
print(test_y)

correct = pred==test_y
print(str(correct.sum()/pred.size*100) + "%")

fig, ax = plt.subplots(figsize=(16,8))
plot_importance(lgbm, ax=ax)
plt.show()