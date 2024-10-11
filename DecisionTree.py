import numpy as np
import pandas as pd

class Node:
    def __init__(self, attribute=None):
        self.attribute = attribute
        self.children = []
        self.children_features = []
        self.answer = ""  # result of classification
        self.node_type = ""
        self.parent = None
        self.depth = 0

    def __str__(self):
        return self.attribute

class DecisionTreeID3:
    def __init__(self):
        self.root = Node()
        self.k_fold_confuse_matrix = []
        self.confuse_matrix = []
        self.not_predict = 0
        self.classes = 0

    def _entropy(self, data: pd.Series):
        N = len(data)
        entropy = 0
        count_data = data.value_counts()
        for i in count_data:
            p = i / N
            entropy += -p * np.log2(p)
        return entropy

    def _gain(self, x_train, y_train, attr, target_entropy):
        unique_values = x_train[attr].unique()
        weighted_entropy = 0
        for value in unique_values:
            new_y_train = y_train[x_train[attr] == value]
            entropy = self._entropy(new_y_train)
            weighted_entropy += len(new_y_train) / len(y_train) * entropy
        return target_entropy - weighted_entropy

    def _learn(self, node, x_train, y_train):
        entropies = []
        target_entropy = self._entropy(y_train)
        for attr in x_train.columns:
            entropies.append(self._gain(x_train, y_train, attr, target_entropy))

        if not entropies:
            node.node_type = "leaf"
            node.answer = y_train.value_counts().idxmax()
            return

        max_index = np.argmax(entropies)
        max_attr = x_train.columns[max_index]
        node.attribute = max_attr
        unique_values = x_train[max_attr].unique()
        for value in unique_values:
            child = Node()
            node.children.append(child)
            node.children_features.append(value)
            child.depth = node.depth + 1
            child.parent = node
            new_x_train = x_train[x_train[max_attr] == value]
            new_y_train = y_train[x_train[max_attr] == value]
            if len(new_y_train.unique()) == 1:
                child.node_type = "leaf"
                child.answer = new_y_train.iloc[0]
            else:
                self._learn(child, new_x_train, new_y_train)

    def _test(self, x_test, y_test):
        self.confuse_matrix = pd.DataFrame(0, index=self.classes, columns=self.classes)

        for i in range(len(x_test)):
            prediction = self.predict(x_test.iloc[[i]])
            actual = y_test.iloc[i]
            if prediction is not None:
                self.confuse_matrix.loc[actual, prediction] += 1
            else:
                self.not_predict += 1
        return self.confuse_matrix

    def fit(self, x_train, y_train, x_test, y_test):
        self.classes = y_train.unique()
        self._learn(self.root, x_train, y_train)
        self.root.node_type = "root"
        self.confuse_matrix = self._test(x_test, y_test)

    def fit_k_fold(self, k_folds):
        for i in range(len(k_folds[0])):
            train_data = pd.concat([k_folds[0][j] for j in range(len(k_folds[0])) if j != i])
            test_data = k_folds[0][i]
            x_train = train_data
            y_train = pd.concat([k_folds[1][j] for j in range(len(k_folds[1])) if j != i])
            x_test = test_data
            y_test = k_folds[1][i]
            self.classes = y_train.unique()
            self._learn(self.root, x_train, y_train)
            self.root.node_type = "root"
            self.k_fold_confuse_matrix.append(self._test(x_test, y_test))

    def predict(self, test_data):
        return self._predict(self.root, test_data)

    def _predict(self, node, test_data):
        if node.node_type == "leaf":
            return node.answer
        else:
            for child, feature in zip(node.children, node.children_features):
                if test_data[node.attribute].iloc[0] == feature:
                    return self._predict(child, test_data)
        return None

    def show_tree(self):
        self._show_tree(self.root)

    def _show_tree(self, node, depth=0):
        if node.node_type == "leaf":
            print(f"{'| ' * depth} {node.answer} (Depth: {node.depth})")
        else:
            print(f"{'| ' * depth} {node.attribute} (Depth: {node.depth})")
            for child, feature in zip(node.children, node.children_features):
                print(f"{'| ' * (depth + 1)} {feature}")
                self._show_tree(child, depth + 2)

    def get_F1(self):
        confuse_matrix = self.confuse_matrix
        precision = []
        recall = []
        for i in range(len(confuse_matrix)):
            tp = confuse_matrix.iloc[i, i]
            fp = np.sum(confuse_matrix.iloc[:, i]) - tp
            fn = np.sum(confuse_matrix.iloc[i, :]) - tp
            precision.append(tp / (tp + fp))
            recall.append(tp / (tp + fn))
        precision = np.mean(precision)
        recall = np.mean(recall)
        return 2 * (precision * recall) / (precision + recall)

    def get_F1_k_fold(self):
        k_fold_confuse_matrix = self.k_fold_confuse_matrix
        f1 = []
        for confuse_matrix in k_fold_confuse_matrix:
            precision = []
            recall = []
            for i in range(len(confuse_matrix)):
                tp = confuse_matrix.iloc[i, i]
                fp = np.sum(confuse_matrix.iloc[:, i]) - tp
                fn = np.sum(confuse_matrix.iloc[i, :]) - tp
                precision.append(tp / (tp + fp))
                recall.append(tp / (tp + fn))
            precision = np.mean(precision)
            recall = np.mean(recall)
            f1.append(2 * (precision * recall) / (precision + recall))
        return np.mean(f1)

    def get_not_predict(self):
        return self.not_predict

    def get_confuse_matrix(self):
        return self.confuse_matrix

    def get_k_fold_confuse_matrix(self):
        return self.k_fold_confuse_matrix
