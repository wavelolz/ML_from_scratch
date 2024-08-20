import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

data = pd.read_csv("titanic.csv")
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1)
data = data.dropna()

le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])
data["Embarked"] = le.fit_transform(data["Embarked"])

X = np.array(data.iloc[:, 1:])
y = np.array(data.iloc[:, 0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(y)

class DecisionTree:
    def __init__(self, left=None, right=None, feature=None, thresh=None, value=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.thresh = thresh
        self.value = value
    
    def calculate_entropy(self, y):
        p = np.bincount(y) / len(y)
        p = p[p>0]
        return -sum(p*np.log2(p))

    def id3(self, X, y):
        best_feature = None
        best_threshold = None
        best_entropy = np.inf
        n_samples, n_features = X.shape

        for feature in np.arange(n_features):
            feature_level = sorted(list(set(X[:, feature])))
            entropy = 0.0
 
            for i in range(len(feature_level)-1):
                med = (feature_level[i]+feature_level[i-1])/2
                left_outputs = y[X[:, feature]<=med]
                right_outputs = y[X[:, feature]>med]

                left_entropy = self.calculate_entropy(left_outputs)
                right_entropy = self.calculate_entropy(right_outputs)

                entropy = (left_entropy*len(left_outputs) + right_entropy*len(right_outputs)) / n_samples
                
                if entropy < best_entropy:
                    best_feature = feature
                    best_threshold = med
                    best_entropy = entropy

        return best_feature, best_threshold

    def build_tree(self, X, y, current_depth=0, max_depth=4):
        y = y.astype(np.int64)

        if current_depth == max_depth or len(X) <= 2 or len(np.unique(y)) == 1:
            return DecisionTree(value=np.bincount(y).argmax())

        feature, thresh = self.id3(X, y)
        
        left_mask = X[:, feature] <= thresh
        right_mask = ~left_mask

        if sum(right_mask) == 0 or sum(left_mask) == 0:
            return DecisionTree(value=np.bincount(y).argmax())

        left = self.build_tree(X[left_mask], y[left_mask], current_depth+1, max_depth)
        right = self.build_tree(X[right_mask], y[right_mask], current_depth+1, max_depth)

        return DecisionTree(left, right, feature, thresh, np.bincount(y).argmax())
     
    def predict(self, x, depth=0):

        if self.left is None and self.right is None:
            return self.value
        
        if x[self.feature] <= self.thresh:
            return self.left.predict(x, depth+1)
        else:
            return self.right.predict(x, depth+1)
    
def print_tree(node, depth=0):
    if node is None:
        return
    print(f"Depth: {depth}, Feature: {node.feature}, Thresh: {node.thresh}, Output: {node.value}")
    print_tree(node.left, depth+1)
    print_tree(node.right, depth+1)


tree = DecisionTree()
tree = tree.build_tree(X_train, y_train)
print_tree(tree)


predict_results = []
for obs in X_test:
    predict_results.append(tree.predict(obs))
accuracy = accuracy_score(y_test, predict_results)
print(accuracy)



        

