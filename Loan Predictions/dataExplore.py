import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

loan_df = pd.read_csv("train.csv") #Reading the dataset in a dataframe using Pandas

#print(loan_df.columns)

#print(loan_df.head(10))
#print(loan_df.describe())

#numeric 
applicantIncome = 'ApplicantIncome'
loanAmount = 'LoanAmount'

#categorical attributes
gender = 'Gender'
married = 'Married'
dependents = 'Dependents'
education = 'Education'
self_employed = 'Self_Employed'
credit_history = 'Credit_History'
property_area = 'Property_Area'



def drawBoxAndHistogram(attribute):

    loan_df[attribute].hist(bins=50)
    plt.show()
    loan_df.boxplot(column=attribute)
    plt.show()

#drawBoxAndHistogram(applicantIncome)

def boxByEducation(attribute):
    loan_df.boxplot(column=attribute, by = 'Education')
    plt.show()

#boxByEducation(loanAmount)



def drawBarChart(attribute, xLabel, yLabel, title):
    value_count = loan_df[attribute].value_counts(ascending=True)
    fig = plt.figure()

    plt.style.use('ggplot')

    x = list(value_count.index)
    

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, value_count, color='green')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)

    plt.xticks(x_pos, x)

    plt.show()


drawBarChart(property_area, property_area, 'Count of Applicants',"Applicants by "+property_area)


def plotHistAndBox(attrName,df):
    df[attrName].hist(bins=50)
    plt.show()
    df.boxplot(column=attrName)
    plt.show()

