import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

X_train = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y_train = dataset.target

clf = LogisticRegression()
clf.fit(X_train, y_train)

with open('model.pickle', 'wb') as fh:
    pickle.dump(clf, fh)

print('successfully dumped pickled model')
