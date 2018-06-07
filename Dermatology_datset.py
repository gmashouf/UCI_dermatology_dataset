#######################################
# Make required packages available
#######################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
########################################
# Import dataset and column name assignment
########################################
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
dataset = pd.read_csv(url, header = None)
header = ["erythema", "scaling", "borders", "itching", "koebner",\
          "polygonal papules", "follicular papules", "oral mucosal",\
          "knee and elbow", "scalp", "history", "melanin incontinence",\
          "eosinophils", "PNL", "fibrosis",\
          "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis", "clubbing",\
          "elongation", "thinning","spongiform pustule", "microabcess",\
          "hypergranulosis", "granular layer", "vacuolisation",\
          "spongiosis", "saw-tooth", "plug", "perifollicular parakeratosis",\
          "inflammatory","band-like", "Age", "Diagnosis"]
diagnosis = ["psoriasis", "seborrheic", "LP", "rosea",\
             "dermatitis", "PRP"]
dataset.columns = header
##########################################
#print of the first 5 rows of data set
##########################################
print (dataset.head())
##############################################
#Taking care of aberrant data
############################################## 
# Removes the rows with nan value
dataset.loc[:, "Age"] = dataset.loc[:, "Age"].replace(to_replace="?", value=float("NaN"))
dataset = dataset.dropna(axis = 0)
#removes outliers
dataset.loc[: , "Age"] = pd.to_numeric(dataset.loc[:, "Age"], errors='coerce')
dataset = dataset[dataset.loc[: , "Age"]  > 0]
##############################################
# Decoding categorical data
##############################################
# decoding all categorical variables
for element in header:
    if (element != "Age") and (element != "history") and (element != "Diagnosis") :
        dataset.loc[dataset.loc[:, element] == 0, element] = "absent"
        dataset.loc[dataset.loc[:, element] == 1, element] = "Low"
        dataset.loc[dataset.loc[:, element] == 2, element] = "Medium"
        dataset.loc[dataset.loc[:, element] == 3, element] = "High"
    elif (element == "history"):
        dataset.loc[dataset.loc[:, element] == 0, element] = "NO"
        dataset.loc[dataset.loc[:, element] == 1, element] = "YES"
    elif (element == "Diagnosis"):
        dataset.loc[dataset.loc[:, element] == 1, element] = diagnosis[0]
        dataset.loc[dataset.loc[:, element] == 2, element] = diagnosis[1]
        dataset.loc[dataset.loc[:, element] == 3, element] = diagnosis[2]
        dataset.loc[dataset.loc[:, element] == 4, element] = diagnosis[3]
        dataset.loc[dataset.loc[:, element] == 5, element] = diagnosis[4]
        dataset.loc[dataset.loc[:, element] == 6, element] = diagnosis[5]
    else:
        pass

###############################################
# Consolidating categories if applicable
###############################################
# checking the number of data in each category to determine possibility of consolidation
for element in header:
    if element != "Age":
        print(dataset.loc[:, element].value_counts())        
dataset.loc[dataset.loc[:, "perifollicular parakeratosis"] == "Low", "perifollicular parakeratosis"] = "Medium"       
dataset.loc[dataset.loc[:, "band-like"] == "Low", "band-like"] = "Medium"
dataset.loc[dataset.loc[:, "plug"] == "Medium", "plug"] = "Low"
dataset.loc[dataset.loc[:, "saw-tooth"] == "Low", "saw-tooth"] = "Medium"
dataset.loc[dataset.loc[:, "vacuolisation"] == "Low", "vacuolisation"] = "Medium"
dataset.loc[dataset.loc[:, "fibrosis"] == "Low", "fibrosis"] = "Medium"
dataset.loc[dataset.loc[:, "eosinophils"] == "Medium","eosinophils" ] = "Low"
dataset.loc[dataset.loc[:, "melanin incontinence"] == "Low", "melanin incontinence"] = "Medium"
dataset.loc[dataset.loc[:, "oral mucosal"] == "Low",  "oral mucosal"] = "Medium"
dataset.loc[dataset.loc[:, "polygonal papules"] == "Low", "polygonal papules"] = "Medium"
#################################################
#One-hot encoding for a categorical column with more than 2 categories
#################################################
for i in header:
    if (i != "Age") and (i != "Diagnosis") and (i != "history") and (i != "eosinophils"):
        for element in dataset.loc[:, i].unique():
            dataset.loc[:, str(i) + str(element)] = (dataset.loc[:, i] == element).astype(int)
# Remove obsolete column
for i in header:
    if (i != "Age") and (i != "Diagnosis") and (i != "history") and (i != "eosinophils"):
        dataset = dataset.drop(i, axis = 1)
# Encodes the categorical variables in diagnosis, history and eosinphils
labelencoder = LabelEncoder()
for i in header:
    if (i == "Diagnosis") or (i == "history") or (i == "eosinophils"):
        dataset.loc[:, i] = labelencoder.fit_transform(dataset.loc[:, i])
# changes the index of outcome column
dataset = dataset.reindex(list([a for a in dataset.columns if a != 'Diagnosis'] + ['Diagnosis']), axis=1)
n = len(dataset.columns) # determines the number of columns in dataset
X = dataset.iloc[:, :(n-1)].values
Y = dataset.iloc[:, (n-1)].values
# Binarize the output
y = label_binarize(Y, classes=[0, 1, 2, 3, 4, 5]) # binerize target column for calculating fpr, tpr and threshold
n_classes = y.shape[1]
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
# Z-normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
################################################
# Logistic Regression
################################################
# predict each class against the other
C_parameter = 50. / len(X_train) # parameter for regularization of the model
class_parameter = 'multinomial' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter
classifier = OneVsRestClassifier(LogisticRegression(C=C_parameter, multi_class=class_parameter,\
                                penalty=penalty_parameter, solver=solver_parameter, tol=tolerance_parameter))        
classifier.fit(X_train, y_train) # Training the algorithm
y_predict = classifier.predict(X_test) # prediction
probas = classifier.predict_proba(X_test) # probability
# Compute ROC curve and ROC area for each class
fpr = dict() # dictionary to assign fpr for each scenario (key is the name of class)  
tpr = dict() # dictionary to assign tpr
roc_auc = dict() #dictionary to assign AUC of each class
th = dict() # dictionary to assign probability threshold
CM = dict()
P = dict()
R = dict ()
F1 = dict()
for i in range(n_classes):
    fpr[i], tpr[i], th [i] = roc_curve(y_test[:, i], probas[:, i]) 
    roc_auc[i] = auc(fpr[i], tpr[i]) # calculated area under the curve for a each scenario
    CM[i] = confusion_matrix (y_test[:, i], y_predict[:, i])
    print ('Confusion matrix (Logistic Regression) for class', i, 'vs other classes: \n', CM[i])
    print ('Logistic Regression | Probability Threshold for class', i, '\n', th[i])
    P[i] = precision_score(y_test[:, i], y_predict[:, i])
    print ("\nPrecision:", np.round(P[i], 3))
    R[i] = recall_score(y_test[:, i], y_predict[:, i])
    print ("\nRecall:", np.round(R[i], 3))
    F1[i] = f1_score(y_test[:, i], y_predict[:, i])
    print ("\nF1 score:", np.round(F1[i], 3))    
plt.figure()
lw = 2
color = ['b', 'r', 'c', 'y', 'm', 'k']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color= color[i],
         lw= lw, label='ROC curve of class  %s (AUC = %s) vs others' %(i, np.round(roc_auc[i], 3)), linestyle = ':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of Logistic Regression')
    plt.legend(loc="lower right")
    plt.show()
#################################################
# GaussianNB
#################################################
# predict each class against the other
classifier = OneVsRestClassifier(GaussianNB())        
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
probas = classifier.predict_proba(X_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
th = dict()
CM = dict()
P = dict()
R = dict ()
F1 = dict()
for i in range(n_classes):
    fpr[i], tpr[i], th [i] = roc_curve(y_test[:, i], probas[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    CM[i] = confusion_matrix (y_test[:, i], y_predict[:, i])
    print ('Confusion matrix (GaussianNB) for class', i, 'vs other classes: \n', CM[i])
    print ('GaussianNB | Probability Threshold for class', i, '\n', th[i])
    P[i] = precision_score(y_test[:, i], y_predict[:, i])
    print ("\nPrecision:", np.round(P[i], 3))
    R[i] = recall_score(y_test[:, i], y_predict[:, i])
    print ("\nRecall:", np.round(R[i], 3))
    F1[i] = f1_score(y_test[:, i], y_predict[:, i])
    print ("\nF1 score:", np.round(F1[i], 3))
plt.figure()
lw = 2
color = ['b', 'r', 'c', 'y', 'm', 'k']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color= color[i],\
             lw= lw, label='ROC curve of class = %s vs others (AUC = %s)' %(i, np.round(roc_auc[i], 3)), linestyle = ':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of GaussianNB')
    plt.legend(loc="lower right")
    plt.show()
###################################################
# BernoulliNB
###################################################
# predict each class against the other
classifier = OneVsRestClassifier(BernoulliNB())        
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
probas = classifier.predict_proba(X_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
th = dict()
P = dict()
R = dict()
F1 = dict()
for i in range(n_classes):
    fpr[i], tpr[i], th [i] = roc_curve(y_test[:, i], probas[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    CM[i] = confusion_matrix (y_test[:, i], y_predict[:, i])
    print ('Confusion matrix (BernoulliNB) for class', i, 'vs other classes: \n', CM[i])
    print ('BernoulliNB | Probability Threshold for class', i, '\n', th[i])
    P[i] = precision_score(y_test[:, i], y_predict[:, i])
    print ("\nPrecision:", np.round(P[i], 3))
    R[i] = recall_score(y_test[:, i], y_predict[:, i])
    print ("\nRecall:", np.round(R[i], 3))
    F1[i] = f1_score(y_test[:, i], y_predict[:, i])
    print ("\nF1 score:", np.round(F1[i], 3))
plt.figure()
lw = 2
color = ['b', 'r', 'c', 'y', 'm', 'k']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color= color[i],
         lw= lw, label='ROC curve of class = %s vs others (AUC = %s)' %(i, np.round(roc_auc[i], 3)), linestyle = ':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of BernoulliNB')
    plt.legend(loc="lower right")
    plt.show()
#################################################
# KNeighborsClassifier
#################################################
# predict each class against the other
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors= n, metric = 'minkowski', p = 1))        
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
probas = classifier.predict_proba(X_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
th = dict()
P = dict()
R = dict()
F1 = dict()
for i in range(n_classes):
    fpr[i], tpr[i], th [i] = roc_curve(y_test[:, i], probas[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    CM[i] = confusion_matrix (y_test[:, i], y_predict[:, i])
    print ('Confusion matrix (KNeighborsClassifier) for class', i, 'vs other classes: \n', CM[i])
    print ('KNeighborsClassifier | Probability Threshold for class', i, '\n', th[i])
    P[i] = precision_score(y_test[:, i], y_predict[:, i])
    print ("\nPrecision:", np.round(P[i], 3))
    R[i] = recall_score(y_test[:, i], y_predict[:, i])
    print ("\nRecall:", np.round(R[i], 3))
    F1[i] = f1_score(y_test[:, i], y_predict[:, i])
    print ("\nF1 score:", np.round(F1[i], 3))
plt.figure()
lw = 2
color = ['b', 'r', 'c', 'y', 'm', 'k']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color= color[i],
         lw= lw, label='ROC curve of class = %s vs others (AUC = %s)' %(i, np.round(roc_auc[i], 3)), linestyle = ':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic KNeighborsClassifie')
    plt.legend(loc="lower right")
    plt.show()