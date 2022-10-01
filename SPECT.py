# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 18:09:14 2022

@author: Tijana
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


def reportStats(TP,TN,FP,FN,klasa):
    print()
    print("klasa " + str(klasa) + ":")
    
    senzitivnost = TP/(TP+FN) #recall
    specificnost = TN/(FP+TN)
    ppv = TP/(TP+FP) #precision
    npv = TN/(TN+FN)
    f1 = 2*(ppv*senzitivnost)/(ppv+senzitivnost)
    acc = (TP+TN)/(TP+FP+TN+FN)
    print("senzitivnost: " + str(round(senzitivnost,2)) + " specificnost: " + str(round(specificnost,2)))
    print("PPV: " + str(round(ppv,2)) + " NPV: " + str(round(npv,2)))
    print("f1 score: " + str(round(f1,2)))
    print("preciznost: " + str(round(acc,2)))

def UporediRezultate(Ytest,Ypredict,method):
    print()
    print("Dobijeni rezultati")
    print(Ypredict)
    print("Trazeni rezultati")
    print(Ytest)

    counter=0;
    
    for i in range (0,len(Ytest)):
        if(Ytest[i]!=Ypredict[i]):
            counter+=1
    #print(counter)
    print("Velicina tening skupa je: " + str(len(Ytrain)))
    print("Velicina test skupa je:" + str(len(Ytest)))
    print("Broj pogodaka: " + str(len(Ytest)-counter))
    print("Broj promasaja: " + str(counter))
    
    title = "confusion matrix : " + method
    
    classes = np.unique(Ytest)
  
    
    fig, ax = plt.subplots()
    cm = metrics.confusion_matrix(Ytest, Ypredict, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    ax.set_yticklabels(labels=classes, rotation=0)
    plt.show()
    
    #klasa 1
    TP=cm[0,0]
    TN=cm[1,1]
    FP=cm[1,0]
    FN=cm[0,1]
    reportStats(TP,TN,FP,FN,1)
    #klasa 2
    TP=cm[1,1]
    TN=cm[0,0]
    FP=cm[0,1]
    FN=cm[1,0]
    reportStats(TP,TN,FP,FN,2)

def LogistickaRegresija(Xtrain,Ytrain,Xtest,Ytest):
    print()
    print("Logisticka regresija: ")
    print()

    lr=LogisticRegression()
    lr.fit(Xtrain,Ytrain)
    
    Ypredict=lr.predict(Xtest)
    method = "Logisticka regresija"
    UporediRezultate(Ytest,Ypredict,method)
    
def StabloOdlucivanja(Xtrain,Ytrain,Xtest,Ytest):
    print()
    print("Stablo odlucivanja: ")
    print()
    
    dtc = DecisionTreeClassifier(min_samples_leaf=15)
    dtc.fit(Xtrain,Ytrain)
    
    Ypredict = dtc.predict(Xtest)
    method = "Stablo odlucivanja"
    UporediRezultate(Ytest,Ypredict,method)
    
def RandomForest(Xtrain,Ytrain,Xtest,Ytest):
    print()
    print("Random forest: ")
    print()
    
    param_grid = {'max_depth':[10,20,30,40,50,None], 'max_features':['auto','sqrt'], 'min_samples_leaf':[1,2,3,4], 'n_estimators':[10,50,100,150]}
    
    rf = RandomForestClassifier()
    rf.fit(Xtrain,Ytrain)
    
    Ypredict = rf.predict(Xtest)
    method = "Random forest"
    UporediRezultate(Ytest,Ypredict,method)    
    
    grid = GridSearchCV(rf, param_grid, cv=5)
    grid.fit(Xtrain,Ytrain)
    
    print()
    print("grid rezultat:")
    print(grid.best_params_)
    print()
    
    
    depth=grid.best_params_.get('max_depth')
    features = grid.best_params_.get('max_features')
    leaf = grid.best_params_.get('min_samples_leaf')
    estim = grid.best_params_.get('n_estimators')
    
    grf = RandomForestClassifier(max_depth=depth, max_features=features, min_samples_leaf=leaf, n_estimators=estim)
    grf.fit(Xtrain,Ytrain)
    Ypredict=grf.predict(Xtest)
    method="Grid, Random forest"
    UporediRezultate(Ytest,Ypredict,method)  
    
def NaivniBajes(Xtrain,Ytrain,Xtest,Ytest):
    print()
    print("Naivni bayes: ")
    print()
    
    gnb = GaussianNB()
    gnb.fit(Xtrain,Ytrain)
    Ypredict = gnb.predict(Xtest)
    method = "Naivni Bayes"
    UporediRezultate(Ytest,Ypredict,method)
    
    
def SupportVectorMachine(Xtrain,Ytrain,Xtest,Ytest):
    print()
    print("Support vector machine: ")
    print()
    
    param_grid={'kernel':['linear','rbf'], 'C':[0.1,0.25,0.5,0.75,1]} #parametri koje koristimo za grid search
    
    svc = SVC()
    svc.fit(Xtrain,Ytrain)
    Ypredict = svc.predict(Xtest)
    method = "Support Vector Machine"
    UporediRezultate(Ytest,Ypredict,method)
    
    grid = GridSearchCV(svc,param_grid,cv=5)
    grid.fit(Xtrain,Ytrain)
    
    print()
    print("grid rezultat:")
    print(grid.best_params_)
    print()
    
    c=grid.best_params_.get('C')
    kernel=grid.best_params_.get('kernel')
    
    gsvc=SVC(C=c,kernel=kernel)
    gsvc.fit(Xtrain,Ytrain)
    Ypredict=gsvc.predict(Xtest)
    method= "Grid, Support vector machine"
    UporediRezultate(Ytest,Ypredict,method)
     
def KNN(Xtrain,Ytrain,Xtest,Ytest):
    print()
    print("K-nearest neighbors: ")
    print()
    
    #hiperparametar je n_neighbors
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(Xtrain,Ytrain)
    
    Ypredict = knn.predict(Xtest)
    method = "KNearestNeighbors"
    UporediRezultate(Ytest,Ypredict,method)


def vizuelizacija(x,y,output,xlabel,ylabel):
    x1=[]
    x2=[]
    y1=[]
    y2=[]
    for i in range(0,len(output)):
        if(output[i]==0):
            x1.append(x[i])
            y1.append(y[i])
        else:
            x2.append(x[i])
            y2.append(y[i])
                 
    a1 = plt.scatter(x1,y1,label='klasa1')
    a2 = plt.scatter(x2,y2,label='klasa2')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend((a1,a2),('klasa1','klasa2'))
    plt.legend(bbox_to_anchor=(1.28,0.9),loc='center right')
    #dodati u legendu
    plt.show()


col_list=["class","F1R","F1S","F2R","F2S","F3R","F3S","F4R","F4S","F5R","F5S","F6R","F6S","F7R","F7S","F8R","F8S","F9R","F9S","F10R","F10S","F11R","F11S","F12R","F12S","F13R","F13S","F14R","F14S","F15R","F15S","F16R","F16S","F17R","F17S","F18R","F18S","F19R","F19S","F20R","F20S","F21R","F21S","F22R","F22S"]

train = pd.read_csv("SPECTFtrain.csv",usecols=col_list)
test = pd.read_csv("SPECTFtest.csv",usecols=col_list)

setPodataka = train.sample(frac=1) 
#izmesala podatke trening skupa

#a = len(setPodataka)

Xcols = train[["F1R","F1S","F2R","F2S","F3R","F3S","F4R","F4S","F5R","F5S","F6R","F6S","F7R","F7S","F8R","F8S","F9R","F9S","F10R","F10S","F11R","F11S","F12R","F12S","F13R","F13S","F14R","F14S","F15R","F15S","F16R","F16S","F17R","F17S","F18R","F18S","F19R","F19S","F20R","F20S","F21R","F21S","F22R","F22S"]]
Ycols = train[["class"]]

Xcols1 = test[["F1R","F1S","F2R","F2S","F3R","F3S","F4R","F4S","F5R","F5S","F6R","F6S","F7R","F7S","F8R","F8S","F9R","F9S","F10R","F10S","F11R","F11S","F12R","F12S","F13R","F13S","F14R","F14S","F15R","F15S","F16R","F16S","F17R","F17S","F18R","F18S","F19R","F19S","F20R","F20S","F21R","F21S","F22R","F22S"]]
Ycols1 = test[["class"]]

Xtrain = Xcols[:]
Ytrain = Ycols[:]
Xtest = Xcols1[:]
Ytest = Ycols1[:]

Ytrain= np.ravel(Ytrain)
Ytest = np.ravel(Ytest)

LogistickaRegresija(Xtrain,Ytrain,Xtest,Ytest)
NaivniBajes(Xtrain,Ytrain,Xtest,Ytest) 
KNN(Xtrain, Ytrain, Xtest, Ytest)
StabloOdlucivanja(Xtrain, Ytrain, Xtest, Ytest)
RandomForest(Xtrain,Ytrain,Xtest,Ytest)
SupportVectorMachine(Xtrain,Ytrain,Xtest,Ytest)

#provere
#print(Xtest)
#print(Xtrain)
#print(Ytrain)
#print(a)
#print(train)
#print(test)
#print(setPodataka)


count = 23
for i in range(1,count):
    string1 = "F"+str(i)+"R"
    string2 = "F"+str(i)+"S"
    input1 = train[[string1]].to_numpy()
    input2 = train[[string2]].to_numpy()
    output = train["class"]
    vizuelizacija(input1,input2,output,string1,string2)

