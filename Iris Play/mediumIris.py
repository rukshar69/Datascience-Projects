
from sklearn.model_selection import train_test_split
from sklearn import tree, neighbors, datasets
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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
logisticRegression_classifier = LogisticRegression(solver='liblinear', multi_class='ovr')
lda_classifier = LinearDiscriminantAnalysis()
gaussian_classifier = GaussianNB()
svc_classifier = SVC(gamma='auto')

splits = [0.1*i for i in range(1,10)]
tree_accuracies = []
neighbor_accuracies = []
logisticRegression_accuracies = []
lda_accuracies = []
gaussian_accuracies = []
svc_accuracies = []

#print(splits)

for i in range(9):
    split = splits[i]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=split)
    #print(len(x_train))



    tree_classifier.fit(x_train,y_train)
    neighbor_classifier.fit(x_train,y_train)
    logisticRegression_classifier.fit(x_train,y_train)
    lda_classifier.fit(x_train,y_train)
    gaussian_classifier.fit(x_train,y_train)
    svc_classifier.fit(x_train,y_train)

    tree_predictions=tree_classifier.predict(x_test)
    neighbor_predictions = neighbor_classifier.predict(x_test)
    logisticRegression_predictions = logisticRegression_classifier.predict(x_test)
    lda_predictions = lda_classifier.predict(x_test)
    gaussian_predictions = gaussian_classifier.predict(x_test)
    svc_predictions = svc_classifier.predict(x_test)

    '''
    print(type(tree_predictions))
    print(len(tree_predictions))
    print(tree_predictions[:10])
    '''

    tree_accuracy = accuracy_score(y_test,tree_predictions)
    neighbor_accuracy = accuracy_score(y_test,neighbor_predictions)
    logisticRegression_accuracy = accuracy_score(y_test, logisticRegression_predictions)
    lda_accuracy = accuracy_score(y_test, lda_predictions)
    gaussian_accuracy = accuracy_score(y_test, gaussian_predictions)
    svc_accuracy = accuracy_score(y_test, svc_predictions)
    
    print("split: ",split)
    print("tree accuracy: ",tree_accuracy)
    print("neighbor accuracy: ",neighbor_accuracy)
    print("log.Regression acc: ",logisticRegression_accuracy)
    print("lda acc: ",lda_accuracy)
    print("gaussian acc: ",gaussian_accuracy)
    print("svc acc: ",svc_accuracy)
    print("-------------------------------")
    
    tree_accuracies.append(tree_accuracy)
    neighbor_accuracies.append(neighbor_accuracy)
    logisticRegression_accuracies.append(logisticRegression_accuracy)
    lda_accuracies.append(lda_accuracy)
    gaussian_accuracies.append(gaussian_accuracy)
    svc_accuracies.append(svc_accuracy)

print(tree_accuracies)
print(neighbor_accuracies)
print(logisticRegression_accuracies)
print(lda_accuracies)
print(gaussian_accuracies)
print(svc_accuracies)

tree_accuracies = np.array(tree_accuracies)
neighbor_accuracies = np.array(neighbor_accuracies)
logisticRegression_accuracies = np.array(logisticRegression_accuracies)
lda_accuracies = np.array(lda_accuracies)
gaussian_accuracies = np.array(gaussian_accuracies)
svc_accuracies = np.array(svc_accuracies)
splits = np.array(splits)

tree_pickle = 'tree_accuracies.pickle'
neighbor_pickle = 'neighbor_accuracies.pickle'
split_pickle = 'splits.pickle'
logRegression_pickle = 'logisticRegression.pickle'
lda_pickle = 'lda.pickle'
gaussian_pickle = 'gaussian.pickle'
svc_pickle = 'svc.pickle'

with open(tree_pickle, 'wb') as f:
    pickle.dump(tree_accuracies, f)

with open(neighbor_pickle, 'wb') as f:
    pickle.dump(neighbor_accuracies, f)

with open(logRegression_pickle, 'wb') as f:
    pickle.dump(logisticRegression_accuracies, f)

with open(lda_pickle, 'wb') as f:
    pickle.dump(lda_accuracies, f)

with open(gaussian_pickle, 'wb') as f:
    pickle.dump(gaussian_accuracies, f)

with open(svc_pickle, 'wb') as f:
    pickle.dump(svc_accuracies, f)

with open(split_pickle, 'wb') as f:
    pickle.dump(splits, f)