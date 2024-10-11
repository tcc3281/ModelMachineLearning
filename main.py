from DecisionTree import DecisionTreeID3
import ProcessFile as pf

def nomal_decision_tree(data):
    train_data, test_data = pf.split_data(data, 0.8)
    tree = DecisionTreeID3()
    x_train = train_data.drop(columns='class')
    y_train = train_data['class']
    x_test = test_data.drop(columns='class')
    y_test = test_data['class']
    tree.fit(x_train, y_train, x_test, y_test)
    print(tree.show_tree())
    print(tree.get_F1())
    print(tree.confuse_matrix)
    print(tree.not_predict)

def k_fold_decision_tree(data):
    k_folds = pf.k_fold_cross_validation(data, 5, 'class')
    tree=DecisionTreeID3()
    tree.fit_k_fold(k_folds)
    print(tree.show_tree())
    print(tree.get_F1_k_fold())
    print(*tree.k_fold_confuse_matrix, sep='\n')
    print(tree.not_predict)

if __name__ == '__main__':
    data = pf.read_csv('data/car_evaluation.csv')
    # nomal_decision_tree(data)
    k_fold_decision_tree(data)