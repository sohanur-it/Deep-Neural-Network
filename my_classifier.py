from scipy.spatial import distance
import matplotlib.pyplot as plt


def euc(a, b):
    return distance.euclidean(a, b)


class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)  # sending row to closest file
            predictions.append(label)

        return predictions

    def closest(self, row):
        # print(row)
        best_dist = euc(row, self.X_train[0])
        # print(best_dist)
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        print("index:", best_index, "predictions:", self.y_train[best_index])

        return self.y_train[best_index]


from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

#print(X, y)
plt.scatter(iris.data)
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#print(X_train.shape, y_train.shape)

#from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

# print(predictions)

# from sklearn.metrics import accuracy_score

# print(accuracy_score(y_test, predictions))
