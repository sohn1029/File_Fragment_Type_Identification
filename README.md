# File Fragment Type Identification

## Introduction
This project was implemented by reading a paper called "FiFTy: Large-scale File Fragment Type Identification Using Convolutional Neural Networks". 
File Fragment Type Identification, which emerged in forensics analysis for finding the file type just by its data block stored in the memory, is being actively studied since 2006. As of 2020, this paper is solving the identification problem using CNN.
This project used 13 features and LightGBM to solve the first scenario(75 file type classifications) that was introduced in the paper.

## Dataset
The original dataset is available at https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset . If you download Scenario 1 dataset with block size 512, you can get '512_1' folder. And let's put it in the same location as 'make_feature.py'. Also, if you do the same for Human-readable labels classes.json, then the dataset is ready.
- x  

![x](https://user-images.githubusercontent.com/31722713/186681391-d938417a-a460-45fc-9f7d-4f9d3ecb6b99.png)

- y  

![y](https://user-images.githubusercontent.com/31722713/186681404-8ed3241e-f1e5-4d83-a6ee-a6b1bfd1d4b7.png)


## Feature Extraction
The paper introduced 14 features. However, 13 features were used in this project, excluding kolmogorov complexity. The feature list (to be added later) is as follows.   
> Arithmetic Mean, Geometric Mean, Harmonic Mean, Standard Deviation, Mean absolute deviation, Hamming Weight, Kurtosis of Byte Value, Skewness, Longest Byte streak, Low ASCII Range Freq, Med ASCII Range Freq, High ASCII Range Freq, Shannon Entropy

Each feature was extracted with features supported by scipy, numpy, and some simple algorithms. In this project, 500000 training data and 100000 test data were conducted, and the same results as this project can be checked, 'make_feature.py' can be executed, and feature_data can be checked. Data and labels are made in the form of csv and npy, respectively.

## Train
When training and test data are prepared by reading the csv file in pandas and reading the npy file in numpy, training and results can be checked with LGBM Classifier supported by lightgbm.
-	Hyperparameters of LGBM Classifier  

![hyperparam](https://user-images.githubusercontent.com/31722713/186679323-49a8e435-b9d6-43ea-a0ef-589c92bc35d3.png)


## Result  

-	Classification accuracy  

![best_result](https://user-images.githubusercontent.com/31722713/186679347-5aea878a-3505-45be-a4f0-14afa7408316.png)

-	Importance for each feature  

![importance](https://user-images.githubusercontent.com/31722713/186679372-0974f605-f8f4-4506-9717-8d61985a8c9c.png)

