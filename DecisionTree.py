import numpy as np
import pandas as pd
from numpy.ma.extras import unique


class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""
        self.node_type = ""

    def __str__(self):
        return self.attribute


class DecisionTreeID3:
    def __init__(self, training_data, target_header):
        self.root = Node("root")
        self.training_data = training_data
        self.target_header = target_header
        self.attributes = list(training_data.columns)

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

    def fit(self):
        self._learn(self.root, self.training_data)
        self.root.node_type = "root"

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
            child = Node(value)
            node.children.append(child)
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
            for child in node.children:
                if test_data[node.attribute].iloc[0] == child.attribute:
                    return self._predict(child, test_data)
        return None

    def show_tree(self):
        self._show_tree(self.root)

    def _show_tree(self, node, depth=0):
        if node.node_type == "leaf":
            print("  " * depth, f"{node.attribute} (Leaf: {node.answer})")
        else:
            print("  " * depth, node.attribute)
        for child in node.children:
            self._show_tree(child, depth + 1)

if __name__ == '__main__':
    data = pd.read_csv("data.csv")
    tree = DecisionTreeID3(data, "play")
    tree.fit()
    tree.show_tree()
