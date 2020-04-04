import pickle
import matplotlib.pyplot as plt

tree_pickle = 'tree_accuracies.pickle'
neighbor_pickle = 'neighbor_accuracies.pickle'
split_pickle = 'splits.pickle'
logRegression_pickle = 'logisticRegression.pickle'
lda_pickle = 'lda.pickle'
gaussian_pickle = 'gaussian.pickle'
svc_pickle = 'svc.pickle'

def loadPickles(tree_pickle, neighbor_pickle, split_pickle,logRegression_pickle,lda_pickle,gaussian_pickle,svc_pickle):

    with open(tree_pickle, 'rb') as f:
        tree_accuracies = pickle.load(f)

    with open(neighbor_pickle, 'rb') as f:
        neighbor_accuracies = pickle.load(f)

    with open(split_pickle, 'rb') as f:
        splits = pickle.load(f)

    with open(logRegression_pickle, 'rb') as f:
        logisticRegression_accuracies = pickle.load(f)

    with open(lda_pickle, 'rb') as f:
        lda_accuracies =pickle.load(f)

    with open(gaussian_pickle, 'rb') as f:
        gaussian_accuracies =pickle.load(f)

    with open(svc_pickle, 'rb') as f:
        svc_accuracies =pickle.load(f)
    return tree_accuracies, neighbor_accuracies, splits, logisticRegression_accuracies, lda_accuracies, gaussian_accuracies, svc_accuracies

tree_accuracies, neighbor_accuracies, splits, logisticRegression_accuracies, lda_accuracies, gaussian_accuracies, svc_accuracies = loadPickles(tree_pickle, neighbor_pickle, split_pickle, logRegression_pickle, lda_pickle,gaussian_pickle, svc_pickle)

def drawNormals():
    plt.plot(splits, tree_accuracies, label = "tree_accuracies")
    plt.plot(splits, neighbor_accuracies, label = "neighbor_accuracies")
    plt.plot(splits, logisticRegression_accuracies, label = "logisticRegression_accuracies")
    plt.plot(splits, lda_accuracies, label = "lda_accuracies")
    plt.plot(splits, gaussian_accuracies, label = "gaussian_accuracies")
    plt.plot(splits, svc_accuracies, label = "svc_accuracies")

    plt.xlabel('test splits')
    # Set the y axis label of the current axis.
    plt.ylabel('accuracies')
    # Set a title of the current axes.
    plt.title('tree and neighbor accuracy')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

#drawNormals()

#PCA Components 2
tree_pickle_pca2 = 'tree_accuracies_pca2.pickle'
neighbor_pickle_pca2 = 'neighbor_accuracies_pca2.pickle'
logRegression_pickle_pca2 = 'logisticRegression_pca2.pickle'
lda_pickle_pca2 = 'lda_pca2.pickle'
gaussian_pickle_pca2 = 'gaussian_pca2.pickle'
svc_pickle_pca2 = 'svc_pca2.pickle'

tree_accuracies, neighbor_accuracies, splits, logisticRegression_accuracies, lda_accuracies, gaussian_accuracies, svc_accuracies = loadPickles(tree_pickle_pca2,neighbor_pickle_pca2,split_pickle,logRegression_pickle_pca2,lda_pickle_pca2,gaussian_pickle_pca2,svc_pickle_pca2)

#drawNormals()

#PCA Components 3

tree_pickle_pca3 = 'tree_accuracies_pca3.pickle'
neighbor_pickle_pca3 = 'neighbor_accuracies_pca3.pickle'
logRegression_pickle_pca3 = 'logisticRegression_pca3.pickle'
lda_pickle_pca3 = 'lda_pca3.pickle'
gaussian_pickle_pca3 = 'gaussian_pca3.pickle'
svc_pickle_pca3 = 'svc_pca3.pickle'

tree_accuracies, neighbor_accuracies, splits, logisticRegression_accuracies, lda_accuracies, gaussian_accuracies, svc_accuracies = loadPickles(tree_pickle_pca3, neighbor_pickle_pca3, split_pickle, logRegression_pickle_pca3, lda_pickle_pca3, gaussian_pickle_pca3, svc_pickle_pca3)
drawNormals()