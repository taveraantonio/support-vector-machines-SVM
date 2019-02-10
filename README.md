# SVM
This code implements Support Vector Machines (SVMs) and perform data validation, cross validation and grid search in order to find the best parameters to evaluate the model.

## What it does
The code, after preparing data, is divided into three different sections: first it deals with the train and the analysis of different C values for a Linear SVM, then it does the same thing but for a RBF kernel and it analyzes the results of a grid search to find the best parameters to setup the model; last it repeats the analysis but with a k-fold cross validation.
An in-depth description of what the code does, is present as comment of the code itself. 
An detailed analysis of the work and of the obtained results is described in the Report file.

## How it works
You can run the code with a Python environment (like Spyder) or directly from the terminal (inside the root folder of the project) like this: 
```
> python /src/SVM_source_code.py
