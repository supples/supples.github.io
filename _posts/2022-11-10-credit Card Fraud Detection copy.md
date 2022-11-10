---
layout: post
title: Credit Card Fraud Detection
subtitle: Kaggle
categories: Kaggle-copy
tags: [test]
---

출처 : https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets



Credit Card Fraud Detection
------------
## Kaggle 필사

다양한 예측 모델을 사용하여 트랜잭션이 정상적인 결제인지 사기인지 탐지하는데 얼마나 정확한지 확인한다. 

### 목표
* 우리에게 제공된 "작은" 데이터의 작은 분포를 이해
* "사기" 및 "비사기" 트랜잭션의 50/50 하위 데이터 프레임 비율을 생성 (NearMiss 알고리즘)
* 사용할 분류기를 결정하고 정확도가 더 높은 분류기를 결정
* 신경망(Neural Network)을 만들고 정확도를 최상의 분류기와 비교
* 불균형 데이터셋으로 인한 일반적인 오류를 이해

### 개요
1. 데이터 이해하기
2. 데이터 전처리
 * Scaling and Distributing
 * Splitting the Data
3. Random UnderSampling and Oversampling
 * Distributing and Correlating
 * Anomaly Detection(이상감지)
 * Dimensionality Reduction and Clustering(t-SNE)
 * Classifiers(분류)
 * SMOTE와 OverSampling
4. Testing
 * Testing with Logistic Regression
 * Neural Networks Testing (UnderSampling vs OverSampling)


### 1. 데이터 이해하기
* 거래 금액은 상대적으로 적고, 모든 마운트의 평균은 대략 USD 88정도 이다. NULL 값이 없으므로 대체할 방법을 찾지 않아도 된다.
* 대부분 트랜잭션은 부정행위(99.83%)가 아닌 반면 부정 행위 트랜잭션은 데이터 프레임에서 시간(0.17%)이 발생한다.


```python

# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('../input/creditcard.csv')
df.head()
```

    /opt/conda/lib/python3.6/site-packages/sklearn/externals/six.py:31: DeprecationWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).
      "(https://pypi.org/project/six/).", DeprecationWarning)



