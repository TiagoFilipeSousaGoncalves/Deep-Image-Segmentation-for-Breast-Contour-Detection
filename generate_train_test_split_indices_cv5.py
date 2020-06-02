# Imports
import _pickle as cPickle
import numpy as np

# Sklearn Imports
from sklearn.model_selection import train_test_split, KFold

# Open X_data to generate the required indices
# The original dataset is already split, but we need to "create" a new one

# Open X_train
with open('data/resized/X_train_221.pickle', 'rb') as f:
    X_train = cPickle.load(f)

# Open X_test
with open('data/resized/X_test_221.pickle', 'rb') as p:
    X_test = cPickle.load(p)

# Concatenate both
X = np.concatenate((X_train, X_test))

# The paper implementation uses 5-Fold Cross-Validation
skf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create temporary lists to append indices
train_indices_list = []
test_indices_list = []

# Iterate through the skf object
for train_indices, test_indices in skf.split(X):
    train_indices_list.append(train_indices)
    test_indices_list.append(test_indices)

# Small sanity check print
print(np.shape(train_indices_list), np.shape(test_indices_list))

# Save files
with open('data/train-test-indices/train_indices_list.pickle', 'wb') as t:
    cPickle.dump(train_indices_list, t, -1)

with open('data/train-test-indices/test_indices_list.pickle', 'wb') as c:
    cPickle.dump(test_indices_list, c, -1)

print('Train and Test Split Finished.')