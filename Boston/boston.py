import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston


def createDataset():
    boston_dataset = load_boston()

    # print(boston_dataset.keys())
    # print(boston_dataset.DESCR)

    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    # print(boston.head())
    boston["MEDV"] = boston_dataset.target
    # print(boston.isnull().sum())
    return boston


boston = createDataset()


def distribution():
    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    sns.distplot(boston["MEDV"], bins=30)
    plt.show()


def heatMap():
    correlation_matrix = boston.corr().round(2)
    # annot = True to print the values inside the square
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.show()


def scatterWithMedv():
    plt.figure(figsize=(20, 5))

    features = ["LSTAT", "RM"]
    target = boston["MEDV"]

    for i, col in enumerate(features):
        plt.subplot(1, len(features), i + 1)
        x = boston[col]
        y = target
        plt.scatter(x, y, marker="o")
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("MEDV")
    plt.show()


def visualization():
    distribution()
    heatMap()
    scatterWithMedv()


def createTrainData(attributeList):
    X = boston[attributeList]
    return X


def linRegAt20PercentUtil():
    attributeList = ["LSTAT", "RM", "PTRATIO", "INDUS", "CRIM", "AGE", "RAD", "ZN", "B"]
    numberOfAttributes = len(attributeList)

    trainRmse = []
    trainR2 = []
    testRmse = []
    testR2 = []

    for i in range(2, numberOfAttributes + 1):
        chosenAttributes = attributeList[:i]
        train_rmse, train_r2, test_rmse, test_r2 = linRegAt20Percent(chosenAttributes)
        trainRmse.append(train_rmse)
        trainR2.append(train_r2)
        testRmse.append(test_rmse)
        testR2.append(test_r2)

    return trainRmse, trainR2, testRmse, testR2


def drawBarChart(resultList, title):
    objects = (
        "2 Features",
        "3 Features",
        "4 Features",
        "5 Features",
        "6 Features",
        "7 Features",
        "8 Features",
        "9 Features",
    )
    y_pos = np.arange(len(objects))
    performance = resultList

    plt.barh(y_pos, performance, align="center", alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel("")
    plt.title(title)

    plt.show()


def linRegAt20Percent(attributeList):
    X = createTrainData(attributeList)
    # X = createTrainData(["LSTAT", "RM", "PTRATIO"])
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS"])
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS" ]) # TAX and INDUS -> .72
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS" ]) # NOX and INDUS -> .76
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS", "CRIM"])
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS", "CRIM", "AGE", "RAD"])
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS", "CRIM", "AGE", "RAD", "ZN"])
    # X = createTrainData( ["LSTAT", "RM", "PTRATIO", "INDUS", "CRIM", "AGE", "RAD", "ZN", "B"])
    # X = createTrainData( ["LSTAT", "RM", "PTRATIO", "INDUS", "CRIM", "AGE", "RAD", "ZN", "B"]) #DIS and AGE -.75
    Y = boston["MEDV"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=5
    )

    """
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    """

    print("------------------------------------------------------------------------")
    print(attributeList)
    print("------------------------------------------------------------------------")

    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)

    # model evaluation for training set
    y_train_predict = lin_model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    train_r2 = r2_score(Y_train, y_train_predict)

    print("The model performance for training set")
    print("--------------------------------------")
    print("RMSE is {}".format(train_rmse))
    print("R2 score is {}".format(train_r2))
    print("\n")

    # model evaluation for testing set
    y_test_predict = lin_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    test_r2 = r2_score(Y_test, y_test_predict)

    print("The model performance for testing set")
    print("--------------------------------------")
    print("RMSE is {}".format(test_rmse))
    print("R2 score is {}".format(test_r2))

    return train_rmse, train_r2, test_rmse, test_r2


def linearRegression():

    X = createTrainData(["LSTAT", "RM"])
    # X = createTrainData(["LSTAT", "RM", "PTRATIO"])
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS"])
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS" ]) # TAX and INDUS -> .72
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS" ]) # NOX and INDUS -> .76
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS", "CRIM"])
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS", "CRIM", "AGE", "RAD"])
    # X = createTrainData(["LSTAT", "RM", "PTRATIO", "INDUS", "CRIM", "AGE", "RAD", "ZN"])
    # X = createTrainData( ["LSTAT", "RM", "PTRATIO", "INDUS", "CRIM", "AGE", "RAD", "ZN", "B"])
    # X = createTrainData( ["LSTAT", "RM", "PTRATIO", "INDUS", "CRIM", "AGE", "RAD", "ZN", "B"]) #DIS and AGE -.75
    Y = boston["MEDV"]

    test_sizes = [0.1 * i for i in range(1, 10)]
    # print(test_sizes)
    trainRmse = []
    trainR2 = []
    testRmse = []
    testR2 = []

    for test_size in test_sizes:

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=5
        )

        """
        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.shape)
        print(Y_test.shape)
        """

        lin_model = LinearRegression()
        lin_model.fit(X_train, Y_train)

        # model evaluation for training set
        y_train_predict = lin_model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(Y_train, y_train_predict))
        r2 = r2_score(Y_train, y_train_predict)
        trainRmse.append(rmse)
        trainR2.append(r2)

        if test_size == 0.2:
            print("The model performance for training set")
            print("--------------------------------------")
            print("RMSE is {}".format(rmse))
            print("R2 score is {}".format(r2))
            print("\n")

        # model evaluation for testing set
        y_test_predict = lin_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, y_test_predict))
        r2 = r2_score(Y_test, y_test_predict)
        testRmse.append(rmse)
        testR2.append(r2)

        if test_size == 0.2:
            print("The model performance for testing set")
            print("--------------------------------------")
            print("RMSE is {}".format(rmse))
            print("R2 score is {}".format(r2))

    return trainRmse, trainR2, testRmse, testR2, test_sizes


# trainRmse, trainR2, testRmse, testR2, test_sizes = linearRegression()
# print(trainRmse, trainR2, testRmse, testR2)


def drawPlot(xValues, y1Values, y2Values, title, ylabel, y1Label, y2Label):
    fig = plt.figure()
    ax = plt.axes()

    x = xValues
    plt.plot(x, y1Values, "-g", label=y1Label)
    plt.plot(x, y2Values, "-r", label=y2Label)

    plt.xlabel("Test Size Ratio")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


"""
drawPlot(
    test_sizes,
    trainRmse,
    testRmse,
    "RMSE for train and test dataset",
    "RMSE",
    "Train",
    "Test",
)


drawPlot(
    test_sizes,
    trainR2,
    testR2,
    "R2 Score for train and test dataset",
    "R2 Score",
    "Train",
    "Test",
)
"""


trainRmse, trainR2, testRmse, testR2 = linRegAt20PercentUtil()

drawBarChart(trainRmse, "Train RMSE when Test Size is 20%")

drawBarChart(testRmse, "Test RMSE when Test Size is 20%")

drawBarChart(trainR2, "Train R2 Score when Test Size is 20%")

drawBarChart(testR2, "Test R2 Score when Test Size is 20%")
