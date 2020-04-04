import pickle
import matplotlib.pyplot as plt


split_pickle = 'splits.pickle'

tree_pickle = 'tree_accuracies.pickle'
neighbor_pickle = 'neighbor_accuracies.pickle'
split_pickle = 'splits.pickle'
logRegression_pickle = 'logisticRegression.pickle'
lda_pickle = 'lda.pickle'
gaussian_pickle = 'gaussian.pickle'
svc_pickle = 'svc.pickle'

tree_pickle_pca2 = 'tree_accuracies_pca2.pickle'
neighbor_pickle_pca2 = 'neighbor_accuracies_pca2.pickle'
logRegression_pickle_pca2 = 'logisticRegression_pca2.pickle'
lda_pickle_pca2 = 'lda_pca2.pickle'
gaussian_pickle_pca2 = 'gaussian_pca2.pickle'
svc_pickle_pca2 = 'svc_pca2.pickle'

tree_pickle_pca3 = 'tree_accuracies_pca3.pickle'
neighbor_pickle_pca3 = 'neighbor_accuracies_pca3.pickle'
logRegression_pickle_pca3 = 'logisticRegression_pca3.pickle'
lda_pickle_pca3 = 'lda_pca3.pickle'
gaussian_pickle_pca3 = 'gaussian_pca3.pickle'
svc_pickle_pca3 = 'svc_pca3.pickle'

def loadPickles(normal_pickle, pca3_pickle, pca2_pickle, split_pickle):

    with open(normal_pickle, 'rb') as f:
        normal_accuracies = pickle.load(f)

    with open(pca3_pickle, 'rb') as f:
        pca3_accuracies = pickle.load(f)

    with open(split_pickle, 'rb') as f:
        splits = pickle.load(f)

    with open(pca2_pickle, 'rb') as f:
        pca2_accuracies = pickle.load(f)
    
    return normal_accuracies, pca3_accuracies, pca2_accuracies,splits


tree_accuracies, tree_accuracies_pca3, tree_accuracies_pca2,splits = loadPickles(tree_pickle, tree_pickle_pca3, tree_pickle_pca2,split_pickle)

def drawCompare(normal_accuracies, pca3_accuracies, pca2_accuracies,splits, string):
    plt.plot(splits, normal_accuracies, label = string +" normal_accuracies")
    plt.plot(splits, pca3_accuracies, label = string +" pca3_accuracies")
    plt.plot(splits, pca2_accuracies, label = string +" pca2_accuracies")
    

    plt.xlabel('test splits')
    # Set the y axis label of the current axis.
    plt.ylabel('accuracies')
    # Set a title of the current axes.
    plt.title('Compare accuracies')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

#drawCompare(tree_accuracies, tree_accuracies_pca3, tree_accuracies_pca2,splits, "tree")


neighbor_accuracies, neighbor_accuracies_pca3, neighbor_accuracies_pca2,splits = loadPickles(neighbor_pickle, neighbor_pickle_pca3, neighbor_pickle_pca2,split_pickle)
logRegression_accuracies, logRegression_accuracies_pca3, logRegression_accuracies_pca2,splits = loadPickles(logRegression_pickle, logRegression_pickle_pca3, logRegression_pickle_pca2,split_pickle)
lda_accuracies, lda_accuracies_pca3, lda_accuracies_pca2,splits = loadPickles(lda_pickle, lda_pickle_pca3, lda_pickle_pca2,split_pickle)
gaussian_accuracies, gaussian_accuracies_pca3, gaussian_accuracies_pca2,splits = loadPickles(gaussian_pickle, gaussian_pickle_pca3, gaussian_pickle_pca2,split_pickle)
svc_accuracies, svc_accuracies_pca3, svc_accuracies_pca2,splits = loadPickles(svc_pickle, svc_pickle_pca3, svc_pickle_pca2,split_pickle)

#drawCompare(neighbor_accuracies, neighbor_accuracies_pca3, neighbor_accuracies_pca2,splits, "neighbor")
#drawCompare(logRegression_accuracies, logRegression_accuracies_pca3, logRegression_accuracies_pca2,splits, "logRegression")
#drawCompare(lda_accuracies, lda_accuracies_pca3, lda_accuracies_pca2,splits, "lda")
#drawCompare(gaussian_accuracies, gaussian_accuracies_pca3, gaussian_accuracies_pca2,splits, "gaussian")
drawCompare(svc_accuracies, svc_accuracies_pca3, svc_accuracies_pca2,splits, "svc")