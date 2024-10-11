from re import search

import numpy as np
import pandas as pd


class Node:
    def __init__(self, attribute=None):
        self.attribute = attribute
        self.children = []
        self.children_features = []
        self.answer = ""  # result of classification
        self.node_type = ""


    def __str__(self):
        return self.attribute


class DecisionTreeID3:
    def __init__(self, target_header, training_data, test_data):
        self.root = Node("root")
        self.training_data = training_data
        self.target_header = target_header
        self.attributes = list(training_data.columns)
        self.k_fold_confuse_matrix = []
        self.confuse_matrix = []
        self.test_data = test_data
        self.not_predict = 0

    def import_data(self, training_data, target_header):
        self.training_data = training_data
        self.target_header = target_header
        self.attributes = list(training_data.columns)

    def _entropy(self, data: pd.Series):
        # Calculate the entropy of a dataset.
        N = len(data)
        entropy = 0
        count_data = data.value_counts()
        for i in count_data:
            p = i / N
            entropy += -p * np.log2(p)
        return entropy

    def fit_k_fold(self, k_folds):
        for i in range(len(k_folds)):
            train_data = pd.concat([k_folds[j] for j in range(len(k_folds)) if j != i])
            self._learn(self.root, train_data)
            self.root.node_type = "root"
            self.k_fold_confuse_matrix.append(self._test(k_folds[i]))



    def fit(self):
        self._learn(self.root, self.training_data)
        self.root.node_type = "root"
        # test
        self.confuse_matrix = self._test(self.test_data)


    def _gain(self, data, attr, target_entropy):
        # Calculate the information gain of a dataset for a given attribute.
        unique_values = data[attr].unique()
        weighted_entropy = 0
        for value in unique_values:
            subset = data[data[attr] == value]
            weighted_entropy += (len(subset) / len(data)) * self._entropy(subset[self.target_header])
        return target_entropy - weighted_entropy

    def _learn(self, node, data):
        entropis = []
        target_entropy = self._entropy(data[self.target_header])
        for attr in self.attributes:
            if attr == self.target_header:
                continue
            entropis.append(self._gain(data, attr, target_entropy))

        if not entropis:
            node.node_type = "leaf"
            node.answer = data[self.target_header].value_counts().idxmax()
            return

        max_index = np.argmax(entropis)
        max_attr = self.attributes[max_index]
        node.attribute = max_attr
        unique_values = data[max_attr].unique()
        for value in unique_values:
            child = Node()
            node.children.append(child)
            node.children_features.append(value)
            child.parent = node
            new_data = data[data[max_attr] == value]
            if self._entropy(new_data[self.target_header]) == 0:
                child.node_type = "leaf"
                child.answer = new_data[self.target_header].iloc[0]
            else:
                new_attributes = self.attributes.copy()
                new_attributes.remove(max_attr)
                self._learn(child, new_data)


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
            print(f"{'| ' * depth} {node.answer}")
        else:
            print(f"{'| ' * depth} {node.attribute}")
            for child, feature in zip(node.children, node.children_features):
                print(f"{'| ' * (depth + 1)} {feature}")
                self._show_tree(child, depth + 2)

    def _test(self, test_data):
        self.confuse_matrix = pd.DataFrame(0, index=self.training_data[self.target_header].unique(),
                                      columns=self.training_data[self.target_header].unique())

        for i in range(len(test_data)):
            prediction = self.predict(test_data.iloc[[i]])
            actual = test_data[self.target_header].iloc[i]
            if prediction is not None:
                self.confuse_matrix.loc[actual, prediction] += 1
            else :
                self.not_predict += 1
        return self.confuse_matrix

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
if __name__ == '__main__':
    data = pd.read_csv("data.csv")
    tree = DecisionTreeID3(data, "play")
    tree.fit()
    tree.show_tree()
