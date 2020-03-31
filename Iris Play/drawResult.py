import pickle
import matplotlib.pyplot as plt

tree_pickle = 'tree_accuracies.pickle'
neighbor_pickle = 'neighbor_accuracies.pickle'
split_pickle = 'splits.pickle'

with open(tree_pickle, 'rb') as f:
    tree_accuracies = pickle.load(f)

with open(neighbor_pickle, 'rb') as f:
    neighbor_accuracies = pickle.load(f)

with open(split_pickle, 'rb') as f:
    splits = pickle.load(f)

'''
print(splits)
print(tree_accuracies)
print(neighbor_accuracies)
'''

plt.plot(splits, tree_accuracies, label = "tree_accuracies")
plt.plot(splits, neighbor_accuracies, label = "neighbor_accuracies")

plt.xlabel('test splits')
# Set the y axis label of the current axis.
plt.ylabel('accuracies')
# Set a title of the current axes.
plt.title('tree and neighbor accuracy')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()