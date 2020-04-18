import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

loan_df = pd.read_csv("train.csv") #Reading the dataset in a dataframe using Pandas

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

#log attributes
loanAmountLog = 'LoanAmount_log'
totalIncomeLog = 'TotalIncome_log'

def determineMissingData():
  tmp =  loan_df.apply(lambda x: sum(x.isnull()),axis=0) 
  sizeDataset = loan_df['ApplicantIncome'].size
  tmp = (tmp/sizeDataset)*100
  return tmp

def drawBarChart(dataFrame, xLabel, yLabel, title): 
    fig = plt.figure()

    plt.style.use('ggplot')

    x = list(dataFrame.index)
    

    x_pos = [i for i, _ in enumerate(x)]

    plt.barh(x_pos, dataFrame, color='green')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)

    plt.yticks(x_pos, x)

    plt.show()

#drawBarChart(determineMissingData(), 'missing data(%)','attributes',  'Missing data in attributes in percentage')

#loan_df.boxplot(column='LoanAmount',by = ['Education','Self_Employed'])
#plt.show()


#=============================================Data IMPUTATION====================================

#imputing Self_Employed
loan_df['Self_Employed'].fillna('No',inplace=True)


table = loan_df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
loan_df['LoanAmount'].fillna(loan_df[loan_df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
#print(table.head())


#normalising loan amount to treat outliers
loan_df['LoanAmount_log'] = np.log(loan_df['LoanAmount'])
'''
loan_df['LoanAmount_log'].hist(bins=20)
plt.show()
loan_df.boxplot(column='LoanAmount_log')
plt.show()
'''

loan_df['TotalIncome'] = loan_df['ApplicantIncome'] + loan_df['CoapplicantIncome']
loan_df['TotalIncome_log'] = np.log(loan_df['TotalIncome'])
#loan_df['TotalIncome_log'].hist(bins=20)
#plt.show() 
#loan_df.boxplot(column='TotalIncome_log')
#plt.show()


loan_df['Gender'].fillna(loan_df['Gender'].mode()[0], inplace=True)
loan_df['Married'].fillna(loan_df['Married'].mode()[0], inplace=True)
loan_df['Dependents'].fillna(loan_df['Dependents'].mode()[0], inplace=True)
loan_df['Loan_Amount_Term'].fillna(loan_df['Loan_Amount_Term'].mode()[0], inplace=True)
loan_df['Credit_History'].fillna(loan_df['Credit_History'].mode()[0], inplace=True)

#=============================================DATA IMPUTATION ENDS==========================

#converting categorical attributes into numerical values for training models
def convertToNumerical():
  var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
  le = LabelEncoder()
  for i in var_mod:
      loan_df[i] = le.fit_transform(loan_df[i])

convertToNumerical()


column_names = ["Accuracy", "Cross_Validation_Score"]
scoresFrame = pd.DataFrame(columns = column_names)

#=============================================CLASSIFICATION BEGINS==========================

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome,model_title):
  global scoresFrame
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(n_splits=5)
  kf.get_n_splits(data)
  error = []
  for train, test in kf.split(data):
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  cross_validation = np.mean(error)
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(cross_validation))


  scoresFrame.loc[model_title] = [ accuracy ,  cross_validation]



outcome_var = 'Loan_Status'

def drawBarChartScores(dataFrame, xLabel, yLabel, title): 
    objects = ('Accuracy', 'Cross_Validation_Score')
    y_pos = np.arange(len(objects))

    # Create an empty list 
    Row_list =[] 
      
    # Iterate over each row 
    for rows in dataFrame.itertuples(): 
        # Create list for the current row 
        my_list =[rows.Accuracy, rows.Cross_Validation_Score] 
          
        # append the list to the final list 
        Row_list.append(my_list) 
      
    # Print the list 
    #print(Row_list) 

    performance = Row_list[0]
    print(performance)

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel(yLabel)
    plt.title(title)

    plt.show()

#predictor_var =  [credit_history,education,married,self_employed,property_area,loanAmountLog,totalIncomeLog,gender,dependents]
'''
#logisticRegression01
model = LogisticRegression()
predictor_var = [credit_history]
classification_model(model, loan_df,predictor_var,outcome_var, "logisticRegression01")
#drawBarChartScores(scoresFrame.loc[['logisticRegression01'],:], 'x','Accuracy','logisticRegression01')

#logisticRegression02
model_name = 'logisticRegression02'
model = LogisticRegression()
predictor_var = [credit_history,education,married,self_employed,property_area]
classification_model(model, loan_df,predictor_var,outcome_var, model_name)
#drawBarChartScores(scoresFrame.loc[[model_name],:], 'x','Accuracy',model_name)



#decision tree 01
model_name = 'DecisionTree01'
model = DecisionTreeClassifier()
predictor_var = [credit_history,'Loan_Amount_Term',loanAmountLog]
classification_model(model, loan_df,predictor_var,outcome_var,model_name)
#drawBarChartScores(scoresFrame.loc[[model_name],:], 'x','Accuracy',model_name)

#decision tree 02
model_name = 'DecisionTree02'
model = DecisionTreeClassifier()
predictor_var = ['Credit_History',loanAmountLog, totalIncomeLog]
classification_model(model, loan_df,predictor_var,outcome_var,model_name)
#drawBarChartScores(scoresFrame.loc[[model_name],:], 'x','Accuracy',model_name)

'''
#All Feature Random Forest
model_name = 'random_forest_all_features'
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area','LoanAmount_log','TotalIncome_log']
classification_model(model, loan_df,predictor_var,outcome_var, model_name)
#drawBarChartScores(scoresFrame.loc[[model_name],:], 'x','Accuracy',model_name)


#Create a series with feature importances:
#featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
#print (type(featimp))
#drawBarChart(featimp, 'importance','attribute names','Feature Importance From Random Forest')


print()
print()
model_name = 'random_forest_top5'
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, loan_df,predictor_var,outcome_var,model_name)
drawBarChartScores(scoresFrame.loc[[model_name],:], 'x','Accuracy',model_name)
