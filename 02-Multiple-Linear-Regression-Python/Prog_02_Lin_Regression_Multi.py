#
# SECTION I. Code information 
#
# This program contains a set of functions corresponding to the 
# MULTIPLE LINEAR REGRESSION
# of the MACHINE LEARNING course offered by STANDFORD UNIVERSITY through COURSERA
# under the instruction of Prof. Andrew Ng.
#
# Coded by Rogelio Cuevas-Saavedra (rogercuevas.ca@gmail.com)
#
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
#     2a. NormalizeFeatures( data ) --->
#		  This function takes "data" as input and computes their
#		  mean and standard devation for their normalization.
#
#     2b. ComputeCost( X, Y, Theta, m ) ---> This function computes the
#         cost function for a given set of (X,Y) data and parameter Theta.
#		  The number of rows in the data set is m.
#                                   
#     2c. GradientDescent( X, Y, Theta, Alpha, NumIter, m ) --->
#		  This function executes a gradient descent for m set of (X,Y) data
#		  with parameters Theta and Alpha.
#
#     2d. NormalEquation( X, Y )---> This function computes the normal equation
#		  for a given (X,Y) set of data
#
#
#     2e. Compute_X_Mu_Sigma() ---> This function computes mean and standard deviation
#								   of a set of data read from a file.
#
#     2f. Multiple_Gradient_Descent(Val_Test,Alphas,MaxNumIter,Iter_min,Iter_max) --->
#         This function executes a set of gradient descent with a set of different Alphas.
#		  This is performed with a maximum number of iterations MaxNumIter and plots
#		  the cost function vs. number of iterations. For each coverged run it computes the
#		  predicted value for a given Val_Test.
#
#	  2g. Call_Normal_Equation(Val_Test) ---> This function determines the normal equation
#         and makes a prediction for a given set of values Val_Test
#
#
#
########################################################################################################################
#
#  SECTION V. Actual code
#
#
#
# We import the standard python modules through sys
import sys
#
# We import the required scientific computing tools through numpy and scipy
#
from numpy import *
##
import scipy
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
#
# We set the directory path
EX_DIRECTORY_PATH = '/Users/rogeliocuevassaavedra/Documents/mlclass/'
#
#
# == Function 2a. ===
#
# This function normalizes data so that they have zero mean
# and standard deviation of one
#
# Arguments: data -- Training set to be normalized
# 
# Local variables: Mu -- Mean of training set
# 				   Sigma -- Standard deviation of training set
#				   Data_Normalized -- Training set after normalization
#
def NormalizeFeatures( data ):
	# Compute mean of data
	Mu 			= mean( data, axis=0 )
	# Shift data by mean
	# Data now have zero mean
	Data_Normalized = data - Mu
	# Compute standard deviation of shifted data
	Sigma 		= std( Data_Normalized, axis=0, ddof=1 )
	# Update data; divide by standard deviation
	# Data now have standard deviation of 1
	Data_Normalized 	= Data_Normalized / Sigma
	# Return computed values
	return Data_Normalized, Mu, Sigma
#
#
#  == Function 2b. ===
#
# This function computes the cost function associated with
# a number "m" of data (X,Y) and parameter Theta.
#
# Arguments: X -- Values X of training set
# 			 Y -- Values Y of training set
#			 Theta -- Parameter of hypothesis function
#			 m -- Number of features
# Local variables: diff -- Difference between predicted and actual
#						   Y values
#
def ComputeCost( X, Y, Theta, m ):
	# Compute difference ...
	diff = X.dot( Theta ) - Y
	# ... and compute from it cost function
	return ( diff.T.dot( diff ) / (2 * m) )[0, 0]
#
#
# == Function 2c. ===
#
# This vectorized function executes gradient descent for 
# a given value of Alpha and certain number of iterations 
# and returns the value of the gradient
# 
# Arguments: X -- Argument of hypothesis function
#            Y -- Y values of the training set 
#            Theta -- Parameter of hypothesis function
#            Alpha -- Given coefficient for gradient descent
#            NumIter -- Given number of iterations
#			 m -- Number of features
#
# Local variables: Grad -- Value of gradient
#                  counter -- Counter to loop over number of iterations
#                  Inner_sum -- Stores cost (error) term in the iteration
#				   J_iterations -- Array keeping track of values of cost 
#								function for every iteration
# 
#
def GradientDescent( X, Y, Theta, Alpha, NumIter, m ):
	# Initialize Gradient with the values of Theta
	Grad  = copy( Theta )
#	max_j 		= shape(X)[1]
	# We initialize J_iterations
	J_iterations 	= []
#	Alpha_div_m = Alpha / m
	#
	# For every interation...
	#
	for counter in range( 0, NumIter ):
		# We store the error between the predicted and actual value 
		# of Y in Inner_sum
		Inner_sum 	= X.T.dot(X.dot( Grad ) - Y)
		#Grad 		= Grad - Alpha_div_m * inner_sum
		# We account for the recently computed error and update
		# the gradient
		Grad 		= Grad - ( Alpha * Inner_sum / m )
		# We store value of cost function for the current interation
		# in J_iterations
		J_iterations.append( ComputeCost(X, Y, Grad, m ) )

	return J_iterations, Grad
#
#  == Function 2d. ===
#
# This function computes the normal equation associated with
# a multi-variable linear regression
#
# Arguments: X -- Argument of hypothesis function
#            Y -- Y values of the training set 
#
def NormalEquation( X, Y ):
	# We just recall the normal equation is given by
	# Inv( Transpose(X) * X) * Transpose(X) * Y
	return linalg.inv(X.T.dot( X )).dot( X.T ).dot( Y )
#
#
#  == Function 2e. ===
#
# This function reads training set from a txt file
# and returns the normalized features along with
# the mean and standard deviation
#
# Arguments: NO arguments
#
# Local variables: X -- X components of training set
#				   Y -- y component of training set
#				   Mu -- Mean 
#				   Sigma -- Standard deviation
#
def Compute_X_Mu_Sigma():
	# Read data from file...
	data = genfromtxt(EX_DIRECTORY_PATH + "/ex1data2.txt", delimiter = ',')
	# Obtain values of independent variables from training set...
	X = data[:, 0:2]
	# ... as well the values of the dependent ones
	Y = data[:, 2:3]
    # Obtain normalized fetures as well as mean and standard deviation
	X, Mu, Sigma = NormalizeFeatures( X )
	# print X
	print 'Mean = ' ,  Mu
	print 'Standard Dev. = ' , Sigma
#
#
#  == Function 2f. ===
#
# This function reads training set from a txt file
# and executes gradient descent for several different
# values of alpha with a given maximum number of iterations...
#
# Arguments: Val_Test -- Set of X values for which the prediction is made.
#            Alphas -- Set of alphas for which grad. descent will be run.
#			 MaxNumIter -- Maximum number of iterations allowed
#			 Iter_min -- Minimum value for the plot cost function vs. iteartions
# 			 Iter_max -- Maximum value for the plot cost function vs. iteartions
#
# Local variables: data -- Data read from TXT file
#				   X -- X components of training set
#				   Y -- y component of training set
#				   m -- Number of rows in the training set
#				   Theta -- Parameters for the linear regression
#				   Mu -- Value of mean
#				   Sigma -- Value of standard deviation
#				   X_test -- A test set of features
#
def Multiple_Gradient_Descent(Val_Test,Alphas,MaxNumIter,Iter_min,Iter_max):
	#
	X_test = Val_Test 
	# We read data from TXT file..
	data = genfromtxt(EX_DIRECTORY_PATH + "/ex1data2.txt", delimiter = ',' )
	# ... and separate accordingly into X and Y
	X = data[:, 0:2]
	Y = data[:, 2:3]
	# We determine number of rows in the data file
	m = shape( X )[0]
	# We normalize features of training set...
	X, Mu, Sigma = NormalizeFeatures( X )
	# and add intercept to X
	X = c_[ ones((m, 1)), X ] 
	# Set up test values of features:
	# The intercept equaling one
	# A surface of 1650.00 ft^2
	# and 3 bedroom 
	#X_test = array([1.0, 1650.0, 3.0])
	# For the test we exclude the intercept
	X_test[1:] = (X_test[1:] - Mu) / Sigma
	# We let the user know the maximum number of iterations...
	print 'Max. number of iterations = ' , MaxNumIter
	# For each Alpha, we do gradient descent and plot the convergence curve
	for Alpha in Alphas:
		# We initialize theta with zeroes
		Theta = zeros( (3, 1) )
		# Compute values of cost function and parameters theta 
		# by invoking the "GradientDescent" function
		J_iterations, Theta = GradientDescent( X, Y, Theta, Alpha, MaxNumIter, m )
		# We let the user know what alpha value we are using...
		print 'Running alpha = ' , Alpha
		# Create an array of number of iterations,
		number_of_iterations = array( [x for x in range( 1,  MaxNumIter + 1 )] ).reshape(  MaxNumIter, 1)
		# plot cost function versus the number of iterations, 
		pyplot.plot( number_of_iterations, J_iterations, '-b' )
		# titled with the current value of alpha being run, and...
		pyplot.title( "Alpha = %f" % (Alpha) )
		# label the plot
		pyplot.xlabel('Number of iterations')
		pyplot.ylabel('Cost J')
		# The X axis of the plot is shown in the given range
		pyplot.xlim( [Iter_min,Iter_max] )
		pyplot.show( block=True )

		# Print converged value for this test...
		print 'Converged value = ' , X_test.dot( Theta )
#
#	
#  == Function 2g. ===
#
#
# This function reads training set from a txt file
# and executes gradient descent for several different
# values of alpha with a given maximum number of iterations...
#
# Arguments: Val_Test -- Set of X values for which the prediction is made.
#
# Local variables: data -- Data read from TXT file
#				   X -- X components of training set
#				   Y -- y component of training set
#				   m -- Number of rows in the training set
#				   Theta -- Parameters for the linear regression
#
def Call_Normal_Equation(Val_Test):
	X_test = Val_Test 
	# Read training set from file
	data = genfromtxt(EX_DIRECTORY_PATH + "/ex1data2.txt", delimiter = ',' )
	# Separate into X and Y accordingly
	X = data[:, 0:2]
	Y = data[:, 2:3]
	# Get number of rows in training set
	m = shape( X )[0]
	# Add intercepts to X
	X = c_[ ones((m, 1)), X ] 
	# Run normal equation to determine Theta from training set
	Theta = NormalEquation( X, Y )
	# Set up test values of features:
	# The intercept equaling one
	# A surface of 1650.00 ft^2
	# and 3 bedroom 
	#X_test = array([1.0, 1650.0, 3.0])
	# Obtain predicted value...
	print 'Predicted value with normal equation = ' , X_test.dot( Theta )


def main():
	set_printoptions(precision=6, linewidth=300)
	print 'Computing mean and standard deviation...'
	Compute_X_Mu_Sigma()
	# We set up the alpha values we will be using,
	Alphas 	= [0.01, 0.03, 0.1, 0.3, 1.0]
	# the maximum number of iterations,
	MaxNumIter = 200
	# and the range of X-axis for the plot to be generated
	Iter_min = 0
	Iter_max = 50
    # Set up test values of features:
	# The intercept equaling one
	# A surface of 1650.00 ft^2
	# and 3 bedroom 
	#  = array([1.0, 1650.0, 3.0])
	Val_Test = array([1.0, 1650.0, 3.0])
	# With the former information we execute gradient descent...
	print 'Executing gradient descent for muliple alphas...'
	Multiple_Gradient_Descent(Val_Test,Alphas,MaxNumIter,Iter_min,Iter_max)
	# Set up test values of features:
	# The intercept equaling one
	# A surface of 1650.00 ft^2
	# and 3 bedroom 
	#  = array([1.0, 1650.0, 3.0])
	Val_Test = array([1.0, 1650.0, 3.0])
	print 'Obtaining normal equation...'
	Call_Normal_Equation(Val_Test)


if __name__ == '__main__':
	main()