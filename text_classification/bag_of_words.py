# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
from sklearn import svm, cross_validation, grid_search
from sklearn.cross_validation import train_test_split
from scipy import sparse

# <codecell>

df_training = pd.read_csv('../handout/training.csv', header=None)
df_training.columns=['name', 'city_code', 'country_code']

# <codecell>

words = list()
for city in df_training.iloc[:,0].values:
    for word in city.split(' '):
        if not word in words:
            words.append(word.lower())

# <codecell>

words = np.array(words)

# <codecell>

features = np.zeros((len(df_training), len(words)))
i = 0
for city in df_training.iloc[:,0].values:
    for word in city.split(' '):
        # Get the index in words
        index = np.where(words == word.lower())
        features[i][index] = 1
    i += 1

# <codecell>

label_country = df_training.iloc[:,2].values
label_country.shape = (label_country.shape[0],1)
label_city = df_training.iloc[:,1].values
label_city.shape = (label_city.shape[0],1)

# <codecell>

data = np.concatenate((features, label_city), axis=1)
data = np.concatenate((data, label_country), axis=1)

# <codecell>

# Save features and dictionary to file
#np.savetxt("training-features-and-labels.csv", data)
#np.savetxt("dictionary.csv", words, fmt="%s")

# <codecell>

# Split into training and validation data
train_data, val_data = train_test_split(data, test_size=0.33)

# <codecell>

# Extract features and labels again from merged data
train_features = train_data[:, :-2]
train_labels_city = train_data[:, -2]
train_labels_country = train_data[:, -1]

train_labels_city.shape = (train_labels_city.shape[0], 1)
train_labels_country.shape = (train_labels_country.shape[0], 1)

val_features = val_data[:, :-2]
val_labels_city = val_data[:, -2]
val_labels_country = val_data[:, -1]

val_labels_city.shape = (val_labels_city.shape[0], 1)
val_labels_country.shape = (val_labels_country.shape[0], 1)

train_all_labels = np.zeros((train_labels_city.shape[0], 1))
for i in range(train_all_labels.shape[0]):
    train_all_labels[i] = str(int(train_labels_city[i][0])) + str(int(train_labels_country[i][0]))

train_all_labels.shape = (train_all_labels.shape[0],1)

val_all_labels = np.zeros((val_labels_city.shape[0], 1))
for i in range(val_all_labels.shape[0]):
    val_all_labels[i] = str(int(val_labels_city[i][0])) + str(int(val_labels_country[i][0]))

val_all_labels.shape = (val_all_labels.shape[0],1)

# <codecell>

# Perform actual training of SVM(s) based on cross validation
#params = {'C': range(1,2)}
clf = svm.LinearSVC()
clf.fit(sparse.csr_matrix(train_features), sparse.csr_matrix(train_all_labels))

print "TRAINING score: {0}".format(clf.score(sparse.csr_matrix(val_features), sparse.csr_matrix(val_all_labels)))

# <codecell>


# <codecell>


