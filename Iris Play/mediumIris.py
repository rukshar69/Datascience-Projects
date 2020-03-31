
from sklearn.model_selection import train_test_split
from sklearn import tree, neighbors, datasets
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

iris=datasets.load_iris()
x=iris.data
y=iris.target

'''
print(len(x[0]))
print(x[0])
print(type(x))
'''

tree_classifier=tree.DecisionTreeClassifier()
neighbor_classifier=neighbors.KNeighborsClassifier()

splits = [0.1*i for i in range(1,10)]
tree_accuracies = []
neighbor_accuracies = []

#print(splits)

for i in range(9):
    split = splits[i]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=split)
    #print(len(x_train))



    tree_classifier.fit(x_train,y_train)
    neighbor_classifier.fit(x_train,y_train)

    tree_predictions=tree_classifier.predict(x_test)
    neighbor_predictions = neighbor_classifier.predict(x_test)

    '''
    print(type(tree_predictions))
    print(len(tree_predictions))
    print(tree_predictions[:10])
    '''

    tree_accuracy = accuracy_score(y_test,tree_predictions)
    neighbor_accuracy = accuracy_score(y_test,neighbor_predictions)
    
    print("split: ",split)
    print("tree accuracy: ",tree_accuracy)
    print("neighbor accuracy: ",neighbor_accuracy)
    print("-------------------------------")
    
    tree_accuracies.append(tree_accuracy)
    neighbor_accuracies.append(neighbor_accuracy)

print(tree_accuracies)
print(neighbor_accuracies)

tree_accuracies = np.array(tree_accuracies)
neighbor_accuracies = np.array(neighbor_accuracies)
splits = np.array(splits)

tree_pickle = 'tree_accuracies.pickle'
neighbor_pickle = 'neighbor_accuracies.pickle'
split_pickle = 'splits.pickle'

with open(tree_pickle, 'wb') as f:
    pickle.dump(tree_accuracies, f)

with open(neighbor_pickle, 'wb') as f:
    pickle.dump(neighbor_accuracies, f)

with open(split_pickle, 'wb') as f:
    pickle.dump(splits, f)