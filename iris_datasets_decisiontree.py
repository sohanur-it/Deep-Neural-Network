from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

#print(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#print(X_train.shape, y_train.shape)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

#print(predictions)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))
