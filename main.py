from DecisionTree import DecisionTreeID3
import ProcessFile as pf

def decision_tree_nomale(data):
    train_data, test_data = pf.split_data(data, 0.8)
    tree = DecisionTreeID3('class',train_data, test_data)
    tree.fit()
    print(tree.get_F1())
    print(tree.confuse_matrix)
    print(tree.not_predict)

def decision_tree_k_fold(data):
    k_folds = pf.k_fold_cross_validation(data, 5)
    tree=DecisionTreeID3('class',data , k_folds)
    tree.fit_k_fold(k_folds)
    print(tree.get_F1_k_fold())
    print(*tree.k_fold_confuse_matrix, sep='\n')
    print(tree.not_predict)

if __name__ == '__main__':
    data = pf.read_csv('data/car_evaluation.csv')
    decision_tree_k_fold(data)