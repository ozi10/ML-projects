# -*- coding: utf-8 -*-
"""Copy of Artificial Neural Network

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18D_7o6lcot2V4UO6f_GbsTjmWUO55k7e

# Artificial Neural Network

### Importing the libraries
"""

import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

"""## Part 1 - Data Preprocessing"""

dataset = pd.read_excel('Folds5x2_pp.xlsx')

dataset.head()

X=dataset.drop('PE', axis=1).values
y=dataset['PE'].values



import seaborn as sns

sns.pairplot(dataset)

"""### Importing the dataset

### Splitting the dataset into the Training set and Test set
"""

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test  =train_test_split(X,y , test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler

Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)

"""## Part 2 - Building the ANN"""

model = tf.keras.Sequential([
                             tf.keras.layers.Dense(32,activation='relu'),
                             tf.keras.layers.Dense(32,activation='relu'),
                             tf.keras.layers.Dense(1)
])

"""## Part 3 - Training the ANN

### Compiling the ANN
"""

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.1) , metrics='mean_absolute_error' , loss= 'mean_squared_error')

"""### Training the ANN model on the Training set"""

model.fit(X_train,y_train,epochs=100,batch_size=32)

"""### Predicting the results of the Test set"""

pred = model.predict(X_test)

pred_df = pd.DataFrame()

pred_df['Actual'] = y_test

pred_df['predicted'] = pred

pred_df.head()

from sklearn.metrics import r2_score,mean_absolute_error

cm = r2_score(y_test,pred)
mae = mean_absolute_error(y_test,pred)
print('R2 Score:',cm)
print('mean absolute error',mae)



