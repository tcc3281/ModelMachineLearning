from DecisionTree import DecisionTreeID3
import ProcessFile as pf

if __name__ == '__main__':
    data = pf.read_csv('data/playtennis.csv')
    tree = DecisionTreeID3(data, "PlayTennis")
    tree.fit()
    tree.show_tree()