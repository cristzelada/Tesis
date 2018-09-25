from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
features = np.array([[-1, -1,0], [-2, -1,-2], [-3, -2,-3], [1, 1,0], [2, 1,-1], [3, 2,1]])

labels = np.array([1, 1, 1, 2, 2, 2])

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.3,
    random_state=42,
)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_test)
print(predictions)