from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np

features = np.array([
    [29, 23, 72],
    [31, 25, 77],
    [31, 27, 82],
    [29, 29, 89],
    [31, 31, 72],
    [29, 33, 77],
])

labels = np.array([
    [0], 
    [1], 
    [1], 
    [0],
    [1],
    [0],
])

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.3,
    random_state=42,
)

clf = tree.DecisionTreeClassifier()
clf.fit(X=X_train, y=y_train)
clf.feature_importances_ # [ 1.,  0.,  0.]
clf.score(X=X_test, y=y_test) # 1.0