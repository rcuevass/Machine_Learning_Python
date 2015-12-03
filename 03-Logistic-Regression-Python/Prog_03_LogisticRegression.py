#
# SECTION I. Code information 
#
# This program contains a set of functions corresponding to 
# LOGISTIC REGRESSION
# of the MACHINE LEARNING course offered by STANDFORD UNIVERSITY through COURSERA
# under the instruction of Prof. Andrew Ng.
#
# Coded by Rogelio Cuevas-Saavedra (rogercuevas.ca@gmail.com)
#
###########################################################################################################################
#
# SECTION II. General purpose of the program
#
# This program reads data from the Ex1_data_02.txt file, performs a 
# LINEAR REGRESSION and plots the given training set as well as the function
# resulting from the linear regression. It also plots the surface and countour plot.
#
#
########################################################################################################################
#
#
#  SECTION III. Documentation for the code.
#
#  1. Imports needed: 
#     
#     (i). sys - to use standard Python modules
#     (ii). numpy - to perform numerical methods calculations
#     (iii). scipy - to perfrom scientific computing
#     (iv). matplotlib - to perform the plotting required
#     (v). mpl_toolkits.mplot3d - to perform 3D plotting
#
#  2. Functions coded:
#
#     2a. PlotPosNeg( data ) --->
#		  This function reads "data" from file and plots them. It shos them
#		  as "positive" and "negative" examples (classification)
#
#     2b. PlotBoundary( data, X, Theta ) ---> This function takes "data", computes
#		  the boundary associated with X and Theta and plots the boundary together with
#		  the training set.
#                                   
#     2c. Sigmoid( x ) --->
#		  This function computes the sigmoid function with argument x.
#
#     2d. ComputeJCost( Theta, X, Y) ---> This function computes the cost function associated
#		  with parameters Theta and the set of data (X,Y)
#
#
#     2e. GradientJCost( X, Y, Theta ) ---> This function computes the gradient of the cost
#		  function associated with parameters Theta and the set of data (X,Y)
#
#     2f. MinimizingTheta( Theta, X, Y, MaxNumIter ) --->
#         This function determines the value of theta that minimizes the cost function
#		  associated with the training set (X,Y) and does so withing  a maximum number
#		  of iterations MaxNumIter
#
#	  2g. ProbPredicted( Theta, X, bin=True ) ---> This function computes the predicted 
#		  probability associated with parameters Theta and a vector X
#
#
#	  2h. PlotTrainingSet() ---> This function does the actual plotting of the training
#		  set by calling the PlotPosNeg( data ) function
#
#	  2i. PlotTrainingSetAndBoundary() ---> This function does the actual plotting of the 
#		  training
#		  set by calling the PlotPosNeg( data ) function
#
#
#
########################################################################################################################
#
#  SECTION V. Actual code
#
# We import the standard python modules through sys
import sys
#
import scipy.optimize, scipy.special
from numpy import *
#
# We import the required plotting tools through matplotlib and mpl_toolkits
#
# To plot in 2D and scaled in cm
#
from matplotlib import pyplot, cm
#
# to perform the 3D plotting
from mpl_toolkits.mplot3d import Axes3D
#
# We set the directory path..
#
EX_DIRECTORY_PATH = '/Users/rogeliocuevassaavedra/Documents/mlclass/'
#
# == Function 2a. ===
#
# This function plots the positive (accepted) and negative (not accepted)
# data of the given set
# 
# Arguments: data -- Given data set
#
# Local variables: None
#
def PlotPosNeg( data ):
	# We separate the given training set into positive (accepted) and negative (not accepted)
	# according to "1" and "0" given in the training set.
	positives  = data[data[:,2] == 1]
	negatives  = data[data[:,2] == 0]
	# We add the X and Y labels...
	pyplot.xlabel("Exam 1 score")
	pyplot.ylabel("Exam 2 score")
	# ... and set the X and Y limit values of the plot.
	pyplot.xlim([30, 105])
	pyplot.ylim([30, 105])
	# We generate the scatter plot according to the separation we just did.
	pyplot.scatter( negatives[:, 0], negatives[:, 1], c='y', marker='o', s=50, linewidths=1, label="Not admitted" )
	pyplot.scatter( positives[:, 0], positives[:, 1], c='r', marker='+', s=50, linewidths=2, label="Admitted" )
	# And add legends
	pyplot.legend()
#
#
# == Function 2b. ===
#
# This function plots the boundary for the training set given.
# 
# Arguments: data -- Given data set
#			 X -- x components of the data given
#			 Theta -- Parameter for the regression 
#
# Local variables: Plot_X -- Set of X values for the plot.
#				   Plot_Y -- Set of Y values computed from the regression base on
#							 the Plot_X array.
#
#
def PlotBoundary( data, X, Theta ):
	# We plot the given training set by invoking the PlotPosNeg(data) function
	PlotPosNeg( data )
	# We determin the X set of data (arguments)...
	Plot_X = array( [min(X[:,1]), max(X[:,1])] )
	# ... and compute the corresponding Y values according to the regression equation
	Plot_Y = (-1./ Theta[2]) * (Theta[1] * Plot_X + Theta[0])
	# We plot the regression line from the just determined set (X,Y)
	pyplot.plot( Plot_X, Plot_Y )
#	
#
# == Function 2c. ===
#
# This function computes the Sigmoid function for a given argument
# 
# Arguments: x -- Argumeny of the Sigmoid function
#			
# Local variables: NONE
#
#
def Sigmoid( x ):
	# We just make use of scipy
	return scipy.special.expit(x) # = 1.0 / (1.0 + exp( -x ))
#
#
# == Function 2d. ===
#
# This function computes the Sigmoid function for a given argument
# 
# Arguments: X -- X component of the training set
#			 Y -- Y component of the training set
# 			 Theta -- Parameter for the regression
#			
# Local variables: m -- Number of rows in the training set
#				   HypoVal -- Variable that stores the value of the
#								   hypothesis function for the argument X 
#								   and parameter Theta.
#				   Aux -- Auxiliary variable that stores the value of the
#						  cost function.
#
def ComputeJCost( Theta, X, Y):
	# We get the number of rows in the training set
	m = shape( X )[0]
	# Compute the value of the hypothesis function by invoking the 
	# Sigmoid function
	HypoVal = Sigmoid(X.dot( Theta ))
	# We obtain the "first" term of the cost function...
	Aux = log( HypoVal ).T.dot( -Y )
	# ... updated by substracting the second term...
	Aux = Aux - log( 1.0 - HypoVal ).T.dot( 1-Y )
	# ... and finally divided by the number of rows (m).
	Aux = Aux / m
	return Aux
	#return Aux.flatten()
#	
#
# == Function 2e. ===
#
# This function computes the value of the gradient of the cost function
#
# Arguments: X -- X component of the trainins set
#			 Y -- Y component of the training set
#			 Theta -- Parameter of the regression
#			
# Local variables: m -- Number of rows in training set
#
#
def GradientJCost( X, Y, Theta ):
	# We determine the number of rows in the training set...
	m = shape(X)[0]
	# ... and return the gradient of the cost function 
	return ( X.T.dot(Sigmoid( X.dot( Theta ) ) - Y)  ) / m
#
#
# == Function 2f. ===
#
# This function determines the value of theta that minimizes the cost function 
# and determines also the minimum value of the cost function.
#
# 
# Arguments: Theta -- Parameters of the sigmoid function
#            X -- X components of the training set
#			 Y -- Y components of the training set
#			 MaxNumIter -- Maxiumum number of iterations
#			
# Local variables: NONE
#
# 
def MinimizingTheta( Theta, X, Y, MaxNumIter ):
	# This function makes use of scipy
	# Minimize a function using the downhill simplex algorithm.
	# This algorithm only uses function values, not derivatives or second derivatives.
	# Reference: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
	result = scipy.optimize.fmin( ComputeJCost, x0=Theta, args=(X, Y), maxiter=MaxNumIter, full_output=True )
	return result[0], result[1]  
#	
#
# == Function 2g. ===
#
# This function computes the probability associated with the predicted value
# performed by the sigmoid function:
# 		If value greater than 0.5 then probability equals 1
#		If value less than or equal to 0.5 then probability equals 0
# 
# Arguments: Theta -- Parameters of the sigmoid function 
#			 X -- X component of the training set
#			 
# Local variables: Prob -- Value of the Sigmoid function evaluated at the parameters and
#							X component
#				   bin -- binary value
#
#
def ProbPredicted( Theta, X, bin=True ):
	Prob = Sigmoid( Theta.dot( X ))
	if bin :
		return 1 if Prob > 0.5 else 0
	else:
		return Prob
#
#	
#
# == Function 2h. ===
#
# This function reads the training set from file and plots it
# 
# Arguments: NONE
#			
# Local variables: data -- data read from file
#				   m -- number of features
#				   n -- number of rows in the training set
#				   X -- X components of training set
#				   Y -- Y components of training set
#
#
def PlotTrainingSet():
	# We read training set from file...
	data = genfromtxt(EX_DIRECTORY_PATH + "/Ex2_data_01.txt", delimiter = ',' )
	# ... and plot it by invoking the PlotPosNeg(data) function
	PlotPosNeg( data )
	pyplot.show()
#
#
# == Function 2i. ===
#
# This function reads the training set from file and plots it. In addition,
# plots the decision boundary and prints out the value of the cost function
# for a given value of theta parameters. It also predicts the probability
# obtained with the Theta obtained from the minimization of the cost function and
# predicts the probability obatined with the minimizing theta for a given test X
# 
# Arguments: NONE
#			
# Local variables: data -- data read from file
#				   m -- number of features
#				   n -- number of rows in the training set
#				   X -- X components of training set
#				   Y -- Y components of training set
#				   Theta -- Array of parameters for the regression
#
#
def PlotTrainingSetAndBoundary():
	# Read training set from file
	data  = genfromtxt(EX_DIRECTORY_PATH + "/Ex2_data_01.txt", delimiter = ',' )
	# Determine number of features and number of rows in file
	m, n  = shape( data )[0], shape(data)[1] - 1
	# Add the extra column of ones (bias) to X
	X 	  = c_[ ones((m, 1)), data[:, :n] ]
	# Determine Y values from data (in this case either 1 or 0)
	Y 	  = data[:, n:n+1]
	# We set Theta to zero,
	Theta = zeros( (n+1, 1) ) 
	# compute the value of the cost function for such a Theta and training set and
	# print it out to screen
	print "Value of cost function J = " ,  ComputeJCost( Theta,X, Y ).flatten()
	# We determine the value of Theta that minimizes the cost function
	Theta, ValCost = MinimizingTheta( Theta, X, Y ,500)	
	print "Value of Theta = " , Theta.T
	print "Value of cost function for minimizing theta = " , ValCost
	PlotBoundary( data, X, Theta )
	pyplot.show()
	# We set the test array,
	Xtest = array([1, 45, 85])
	# print its value and..
	print "Test array = " , Xtest[1:]
	# compute the predicted probability
	print "Predicted probability = " , ProbPredicted( Xtest, Theta )

def main():
	# We set printing options
	set_printoptions(precision=6, linewidth=200)
	print "Plotting trainig set..."
	PlotTrainingSet()
	print "Printing training set and boundary..."
	PlotTrainingSetAndBoundary()
	

if __name__ == '__main__':
	main()