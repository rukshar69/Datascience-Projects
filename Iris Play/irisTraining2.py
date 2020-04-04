from sklearn.model_selection import train_test_split
from sklearn import tree, neighbors, datasets
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


iris=datasets.load_iris()
x=iris.data
y=iris.target

transposedDb= x.transpose()

#print(transposedDb.shape)
pdDataset = pd.DataFrame(data=x,index=[i for i in range(150)], columns=["sepal-length", "sepal-width", "petal-length", "petal-width"])
#print(pdDataset.head())
pdDataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# histograms
pdDataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(pdDataset)
plt.show()

