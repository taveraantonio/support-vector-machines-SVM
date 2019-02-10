#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI&ML Homework - Prof. Caputo, Year 2018
Homework 2
@student: Antonio Tavera 
@id: 243869

Created on Sun Nov 25 16:07:25 2018
"""
import os, sys 
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import itertools

# Variables 
X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []
C = []
target_names = []
gamma = [10**-11, 10**-9, 10**-7, 10**-5, 10**-3, 10**-1, 10]



###############   Load the dataset  ################
# The function load the iris dataset from the scikit
# learn library. It selects only the first two 
# dimensions and spit the dataset into training, 
# validation and test set according to the received 
# proportion 
# 
# Returns X_train, X_va, X_test, y_train, y_val, 
# y_test and the target names
#
#####################################################
def load_dataset(validation_size, test_size):
	# Load iris dataset 
	iris = load_iris()
	# Select the first two dimensions
	X = iris.data[:, :2]
	# Load labels and target names
	y = iris.target
	target_name = iris.target_names
	# Split dataset into training and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
	# Split training set into training and validation set
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=None)

	return X_train, X_val, X_test, y_train, y_val, y_test, target_name; 


###########  Create and train the model #############
# Create an SVM model by setting C and gamma if they
# are received as input variables. Then it trains the
# model with the received X_train and y_train 
#
# Returns the trained model 
#
#####################################################
def train(X_train, y_train, kernel, c, gamma=None):
	#create the model with the specified kernel and gamma 
	if gamma == None:
		model = svm.SVC(kernel=kernel, C=c)
	else:
		model = svm.SVC(kernel=kernel, C=c, gamma=gamma)
	#train the model 
	model.fit(X_train, y_train)
	
	return model 
	 

################  Test the model  ###################
# Receives the trained model, the X_test dataset and 
# its labels. Make a prediction on the model and 
# compute and print the accuracy. 
#
# Returns the accuracy
# 
#####################################################
def test_model(model, X_test, y_test):
	# Make a prediction on the test set
	test_predictions = model.predict(X_test)
	test_predictions = np.round(test_predictions)
	# Report the accuracy of that prediction 
	accuracy = accuracy_score(y_test, test_predictions)

	return accuracy 


############  Plot Decision Boundaries ################
# Receives the train data, the model, the title to give 
# to the figure and the folder where to save the figure
# It firsts creates a mesh of point to plot in, then it 
# predicts the decision boundaries for the classifier, 
# plot the contourf and the data as data points. It 
# creates the folder where to save the image if not 
# already exists and procede to save the image inside
# 
#####################################################
def plot_boundaries(X_train, y_train, model, title, folder):
	# Create a mesh of point to plot in
	x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
	y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
	# Start plotting the figure
	plt.figure()
	# Predict the decision boundaries for the classifier
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	# Plot contours
	plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
	# Plot points
	plot_colors = "ryb"
	for i, color in zip(np.unique(y_train), plot_colors):
		idx = np.where(y_train == i)
		plt.scatter(X_train[idx, 0], X_train[idx, 1], c=color, 
			  cmap=plt.cm.RdYlBu, label=target_names[i], s=15, edgecolors='black')
	# Set labels and titles
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.title(title)
	# Create folder to save the created image inside
	if not os.path.exists(folder):
		os.makedirs(folder)
	# Save figure inside the specified folder and close plt
	plt.legend()
	plt.savefig(folder + "/" + title + ".jpg")
	plt.close()
	
	return




##################    MAIN     ######################
####################  Menu  #########################
# 1.Train a Linear SVM, plot the decision boundaries,
#	evaluate the method on the validation set, plot 
#	a graph showing the accuracy varying C, evaluate 
#	the model on the test set using the best C value.
#
# 2.Do the same as the point 1 but using a non linear 
#	kernel, RBF kernel. Performs a grid search of the 
#	best parameters for an RBF kernel. Evaluates the 
#	best parameters on the test set and plot 
#	boundaries
#
# 3.Merge training and validation set, Repeat the grid 
#	search for gamma and C but this time perform 5-fold 
#	validation. Evaluate the parameters on the test set.
#
# 4.Exit the process
######################################################	
while(1):
	print('\n')
	print("###### MENU ######")
	user_choice = str(input(
				'1. Linear SVM\n' + \
				'2. RBF Kernel\n' + \
				'3. K-Fold\n' + \
				'4. Exit\n' + \
				'Input: '))
	print("##################")


	if user_choice == '1':
		# LINEAR SVM 
		
		# Load the dataset if not already done 
		if(len(X_val) == 0):
			X_train, X_val, X_test, y_train, y_val, y_test, target_names = load_dataset(0.2, 0.3); 
		
		# Create array of C value parameters
		accuracies=[]
		if(len(C) == 0):
			C.append(10**-3)
			for i in range(0, 6): 
				C.append(C[i]*10)
		
		for c in C:
			# a)Train a linear svm on the training set 
			model = train(X_train, y_train, "linear", c)
			# b)Plot the data and the decision boundaries 
			plot_boundaries(X_train, y_train, model, "Linear SVM with C=" + str(c), "BuondariesLinearSVM")
			# c)Evaluate method on the validation set 
			accuracies.append(test_model(model, X_val, y_val))
		
		# Show how validation accuracy change when we modify C
		print('\n'+'How validation accuracy change modifying C:')
		for c, score in zip(C, accuracies):
			print("C: %.3f,\t score: %.4f" %(c, score))
		print('Best score %.4f' %(np.amax(accuracies)))
		
		# Plot the data above on a graph
		plt.axes(xscale='log')
		plt.plot(C, accuracies, marker='o')
		plt.show()

		# Use the best value of C and evaluate the model on the test set.
		# There will be more than one equal C value
		best_accuracies = np.argwhere(accuracies == np.amax(accuracies))
		best_accuracy = 0
		for best in best_accuracies: 
			# Train a linear svm on the training set with the best c value 
			model = train(X_train, y_train, "linear", C[best[0]])
			# Evaluate model on the test set
			accuracy = test_model(model, X_test, y_test)
			# Update best accuracy
			if(accuracy > best_accuracy):
				best_accuracy = accuracy
				c = C[best[0]]		
		print ("Best C: %.4f with accuracy on test set: %.4f"  %(c, best_accuracy))
			
	
	
	elif user_choice == '2':
		#RBF KERNEL
		
		# Load the dataset if not already done 
		if(len(X_val) == 0):
			X_train, X_val, X_test, y_train, y_val, y_test, target_names = load_dataset(0.2, 0.3);
		
		# Create array of C value parameters if not already done
		accuracies=[]
		if(len(C) == 0):
			C.append(10**-3)
			for i in range(0, 6): 
				C.append(C[i]*10)
		
		for c in C:
			# a)Train a rbf kernel on the training set 
			model = train(X_train, y_train, "rbf", c, "auto")
			# b)Plot the data and the decision boundaries 
			plot_boundaries(X_train, y_train, model, "RBF kernel with C=" + str(c), "BuondariesRBF")
			# c)Evaluate method on the validation set 
			accuracies.append(test_model(model, X_val, y_val))
		
		# Show how validation accuracy change when we modify C
		print('\n'+'How validation accuracy change modifying C:')
		for c, score in zip(C, accuracies):
			print("C: %.3f,\t score: %.4f" %(c, score))
		print('Best score %.4f' %(np.amax(accuracies)))
		
		# Plot the data above on a graph
		plt.axes(xscale='log')
		plt.plot(C, accuracies, marker='o')
		plt.show()

		# Use the best value of C and evaluate the model on the test set.
		# There will be more than one equal C value
		best_accuracies = np.argwhere(accuracies == np.amax(accuracies))
		best_accuracy = 0
		for best in best_accuracies: 
			# Train a svm with rbf kernel on the training set with the best c value 
			model = train(X_train, y_train, "rbf", C[best[0]], "auto")
			# Evaluate model on the test set
			accuracy = test_model(model, X_test, y_test)
			# Update best accuracy
			if(accuracy > best_accuracy):
				best_accuracy = accuracy
				c = C[best[0]]
		print("Accuracy on the test set with best C: %.4f is: %.4f"  %(c, best_accuracy))
		
		
		# Perform a Grid Search of the best parameters for an RBF kernel setting c and gamma
		print("\nPerforming Grid Search of the best parameters")
		accuracies=[]
		for c, g in list(itertools.product(C, gamma)) :
			# Train a rbf kernel on the training set 
			model = train(X_train, y_train, "rbf", c, g)
			# Evaluate method on the validation set 
			accuracies.append(test_model(model, X_val, y_val))
			#print("C: %.1E, G: %.1E, accuracy: %.2f" %(c, g,test_model(model, X_val, y_val) ))
			
		# Save results in a pandas Dataframe 
		gridSearchTable = pd.DataFrame(np.array(accuracies).reshape(len(C),len(gamma)), np.array(C), np.array(gamma))
		# Save the table into a .csv file
		gridSearchTable.to_csv("GridSearchTable.csv")
		# Search for the best C and Gamma
		npa = gridSearchTable.values
		maxCIndex, maxGIndex = np.where(npa == np.amax(npa))
		
		for c, g in zip(maxCIndex, maxGIndex):
			maxC = C[c]
			maxGamma = gamma[g]
			print("Max C value: %.4f and max Gamma: %.2E" %(maxC, maxGamma))
			# Train the model with the best parameters
			model = train(X_train, y_train, "rbf", maxC, maxGamma)
			# Evaluate model on the test set 
			accuracy = test_model(model, X_test, y_test)
			print("Accuracy on the test set with best parameters is: %.3f" %(accuracy))
			# Plot the decision boundaries
			plot_boundaries(X_train, y_train, model, "RBF kernel with best C="+ str(maxC)+ " and gamma=" + str(maxGamma), "GridSearchRBF")
		

	elif user_choice == '3':
		# K-FOLD
		
		# Load the dataset if not already done 
		if(len(X_val) == 0):
			X_train, X_val, X_test, y_train, y_val, y_test, target_names = load_dataset(0.2, 0.3);
		
		# Create array of C value parameters if not already done
		accuracies=[]
		if(len(C) == 0):
			C.append(10**-3)
			for i in range(0, 6): 
				C.append(C[i]*10)
		
		# Merge Training and Validation set 
		X_train = np.concatenate((X_train, X_val))
		y_train = np.concatenate((y_train, y_val))
		X_val = []
		# Perform a Grid Search for gamma and c with 5-fold validation 
		parameters = dict(gamma=gamma, C=C)
		svc = svm.SVC(kernel='rbf')
		grid = GridSearchCV(svc, param_grid=parameters, cv=5, iid=True)
		# Fit the model 
		grid.fit(X_train, y_train)
		# Print the best params and score
		print("Best parameters after gridSearch with 5-fold validation are:\n%s" %(grid.best_params_))
		print("Best score is: %0.2f " %(grid.best_score_))
		# Evaluate the best parameters on the test set
		grid.predict(X_test)
		score = grid.score(X_test, y_test)
		print("Accuracy on the test set with the best parameters is: %0.2f " %(score))
		
		
	elif user_choice == '4': 
		# Exit
		sys.exit()
	else:
		print('You made a not correct choice. Try again!')
		
