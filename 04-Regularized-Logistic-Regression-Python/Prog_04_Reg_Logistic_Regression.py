#
# SECTION I. Code information 
#
# This program contains a set of functions corresponding to 
# REGULARIZED LOGISTIC REGRESSION
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
# This program reads data from the Ex2_data_02.txt file, performs a 
# REGULARIZED LOGISTIC REGRESSION and plots the given training set as well as the 
# decision line obtained from the regression. 
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
#     2b. Sigmoid( x ) --->
#		  This function computes the Sigmoid function with argument x.
#
#
#	  2c. ComputeJCost( Theta, X, Y, Lambda ) --->
#		  This function computes the cost
#		  function associated with parameters Theta, set of data (X,Y) and 
#		  regularization parameter Lambda
#
#	  2d. GradJCost( Theta, X, Y, Lambda ) --->
#		  This function computes the gradient of the cost function with
#		  parameters Theta, set of data (X,Y) and 
#		  regularization parameter Lambda
#
#	  2e. MinimizingTheta( Theta, X, Y, Lambda ) --->
#		  This function minimizes the cost function with
#		  parameters Theta, set of data (X,Y) and 
#		  regularization parameter Lambda
#
#	  2f. MonomialsFeatures( X1, X2 ) ---> 
# 		  This function generates all the monomials of certain degree 
#		  that are obtained from the training set.
#
#	  2g. PlotTrainingSet() --->
#         This function plots the training set and does so by calling the 
#         function PlotPosNeg( data )
#     
#     2h. PrintMonomialsArray() --->
#         This function generates the array of monomials and prints it out 
#         to screen.
#    
#     2i. ComputeCostFunction(Lambda) --->
#         This function reads the training set from file and computes the 
#		  cost function for a given Lambda, minimizes it and prints out 
#         the minimizing theta and cost resulting from the minimization.
#
#     2j. PlotBoundaries() --->
#         This function plots decisions boundaries in addition to the 
#		  plotting of the given training set.
#
#
#
########################################################################################################################
#
#  SECTION V. Actual code
#
# We import the standard python modules through sys
#
import sys
import scipy.optimize, scipy.special
from numpy import *
#
# We import the required plotting tools through matplotlib and mpl_toolkits
#
# To plot in 2D and scaled in cm
#
from matplotlib import pyplot, cm
#
# We set the directroy path...
#
EX_DIRECTORY_PATH = '/Users/rogeliocuevassaavedra/Documents/mlclass/'
#
#
# == Function 2a. ===
#
# This function plots the positive (accepted) and negative (not accepted)
# data of the given set
# 
# Arguments: data -- Given data set
#
# Local variables: None

def PlotPosNeg( data ):
	# We separate the given training set into positive (accepted) and negative (not accepted)
	# according to "1" and "0" given in the training set.
	negatives = data[data[:, 2] == 0]
	positives = data[data[:, 2] == 1]
	# We add the X and Y labels to the plot...
	pyplot.xlabel("Microchip test 1")
	pyplot.ylabel("Microchip test 2")
	# ... and set the X and Y limit values of the plot.
	pyplot.xlim([-1.0, 1.4])
	pyplot.ylim([-1.0, 1.4])
	# We generate the scatter plot according to the separation we just did,
	pyplot.scatter( negatives[:,0], negatives[:,1], c='g', marker='o', linewidths=1, s=40, label='y=0' )
	pyplot.scatter( positives[:,0], positives[:,1], c='r', marker='x', linewidths=2, s=40, label='y=1' )
	# and add legends to it.
	pyplot.legend()
#
#
# == Function 2b. ===
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
# == Function 2c. ===
#
# This function computes the cost function for the following arguments:
# 
# Arguments: Theta -- Parameter for the regression
#			 X -- X component of the training set
#			 Y -- Y component of the training set
#			 Lambda -- Regularization parameter
# 			 
#			
# Local variables: m -- Number of data points in the training set
#				   HypoVal -- Variable that stores the value of the
#								   hypothesis function for the argument X 
#								   and parameter Theta.
#				   Aux1 and Aux2 -- Auxiliary variables that stores the 
#									non-regularized contribution of the cost function
#				   NonRegTerm -- Non-regularized contribution of the cost function
#				   RegTerm -- Regularized contribution of the cost function
#
#
def ComputeJCost( Theta, X, Y, Lambda ):
	# We get the number of data points in the training set
	m = shape( X )[0]
	# Then we obtain the value of the hypothesis function
	HypoVal 	   = Sigmoid( X.dot( Theta ) )
	# The non-regularized term in the regression has two contributions. We
	# denote such contributions as Aux1 and Aux2 and compute them:
	Aux1 	   = log( HypoVal ).dot( -Y )
	Aux2 	   = log( 1.0 - HypoVal ).dot( 1 - Y )
	# From the auxiliary terms we obtain the non-regularized contribution
	# to the cost function:
	NonRegTerm = (Aux1 - Aux2) / m
	# And then we compute the regularized contribution:
	RegTerm = Theta.transpose().dot( Theta ) * Lambda / (2*m)
	# We finally determine the cost function:
	return NonRegTerm + RegTerm
#
#	
#
# == Function 2d. ===
#
# This function computes the value of the gradient of the cost function
#
# Arguments: Theta -- Parameter of the regression
#			 X -- X component of the trainins set
#			 Y -- Y component of the training set
#			 Lambda -- Regularization parameter for the regression
#			 
#			
# Local variables: m -- Number of rows in training set
#				   Grad -- Value of the gradient
#
def GradJCost( Theta, X, Y, Lambda ):
	# We get the number of data points in the training set
	m = shape( X )[0]
	# We compute the gradient of the non-regularized term of the cost function,
	Grad = X.T.dot( Sigmoid( X.dot( Theta ) ) - Y ) / m
	# and then updated it with the gradient of the regularized term of the 
	# cost function: ( (Theta[1:] * Lambda ) / m )
	Grad[1:] = Grad[1:] + ( (Theta[1:] * Lambda ) / m )
	return Grad
#
#
# == Function 2e. ===
#
# This function minimizes the cost function and determines the minimizing
# theta
#
# Arguments: Theta -- Parameter of the regression
#			 X -- X component of the trainins set
#			 Y -- Y component of the training set
#			 Lambda -- Regularization parameter for the regression
#			 
#			
# Local variables: m -- Number of rows in training set
#				   Grad -- Value of the gradient
#
def MinimizingTheta( Theta, X, Y, Lambda ):
	# We simply make use of scipy to minimize the cost function.
	# SciPy permits minimizing a scalar function of one or more
	# variables.
	# The following is a link to the reference documentation to 
	# scipy.optimize.minimize
	# http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
	result = scipy.optimize.minimize( ComputeJCost, Theta, args=(X, Y, Lambda),  method='BFGS', options={"maxiter":500, "disp":True} )
	# Finally obtain the minimizing argument and minimum value of the function
	# subject to minimization.
	return result.x, result.fun
#
#
# == Function 2f. ===
#
# This function generates all the monomials of certain degree that are obtained
# from the training set. The resulting monomials are then arranged in a column vector.
#
# Arguments: X1 -- X1 coordinate of the training set
#			 X2 -- X2 coordinate of the training set
#			
# Local variables: MaxDegree -- Maximum degree of the monomials to be generated
#				   MonomialsVector -- Column vector that stores the generated monomials
#				   i, j -- Just counters
#				   X1power -- Variable temporarily storing powers of X1
#				   X2power -- Variable temporarily storing powers of X2
#				   Monomials -- Variable temporarily storing monomials
#					
#									 
def MonomialsFeatures( X1, X2 ):
	# We set the maximum degree of monomials to be generated 
	MaxDegree = 6
	# We first set the resulting MonomialsVector with a vector of ones
	# and length the number of points in the training set.
	MonomialsVector = ones( (shape(X1)[0], 1) )
	# We now generate all the possible monomials from degree zero to MaxDegree
	# as X^(i-j)*Y^j. This implies having two counters:
	# i which controls the power of X coordinate of training set
	for i in range(1, MaxDegree+1):
		# and j which controls the power of Y coordinate of training set
		for j in range(0, i+1):
			# The just gerenated powers are stored in temporal variables...
			X1power = X1 ** (i-j)
			X2power = X2 ** (j)
			# Monomials are formed as the aformentioned products
			Monomials  = (X1power * X2power).reshape( shape(X1power)[0], 1 ) 
			# We now update MonomialsVector as a horizontal stackup of the 
			# originally created vector of ones and the recently created 
			# vector of Monomials
			MonomialsVector   = hstack(( MonomialsVector, Monomials ))
	# We return the final array
	return MonomialsVector
#
#
# == Function 2g. ===
#
# This function plots the training set and does so by calling the function
# PlotPosNeg( data )
#
# Arguments: None
#			
# Local variables: data -- Training set to be plotted
#				   
#					
def PlotTrainingSet():
	# We read data from file...
	data  = genfromtxt(EX_DIRECTORY_PATH + "/Ex2_data_02.txt", delimiter = ',' )
	# and plot them using the previously coded function
	PlotPosNeg( data )
	pyplot.show()
#
#
# == Function 2h. ===
#
# This function generates the array of monomials and prints it out to screen.
#
# Arguments: None
#			
# Local variables: data -- Training set to be plotted
#				   Xmon -- Array used to store the monomials
#				   
# IMPORTANT NOTE. This function only prints out the array and monomials and its calling
# 				  does not affect the general purpouse of this program. It can therefore
#				  be commented in the main body of the code.
#					
def PrintMonomialsArray():
	data  = genfromtxt(EX_DIRECTORY_PATH + "/Ex2_data_02.txt", delimiter = ',' )
	Xmon  = MonomialsFeatures( data[:, 0], data[:, 1] )
	print Xmon


#
# == Function 2i. ===
#
# This function reads the training set from file and computes the cost function
# for a given Lambda, minimizes it and prints out the minimizing theta
# and cost resulting from the minimization.
#
# Arguments: Lambda
#			
# Local variables: data -- Training set to be read from file
#				   Y -- Y component of training set
#				   X -- X component of training set
#				   Theta -- Parameter of the regression
# 				   CostValue -- Value of cost function
#				   cost -- Minimized value of the cost function
#				   
# IMPORTANT NOTE. This function only prints out the array and monomials and its calling
# 				  does not affect the general purpouse of this program.
#					

def ComputeCostFunction(Lambda):
	data  = genfromtxt(EX_DIRECTORY_PATH + "/Ex2_data_02.txt", delimiter = ',' )
	Y 	  = data[:,2]
	X 	  = MonomialsFeatures( data[:, 0], data[:, 1] )
	Theta = zeros( shape(X)[1] )
	CostValue = ComputeJCost( Theta, X, Y, Lambda )
	print "Value of cost function: " , CostValue
	Theta, cost = MinimizingTheta( Theta, X, Y, Lambda )
	print "For lambda equals " , Lambda 
	print "the minimized cost function is " , cost
	print "and the minimizing theta is " 
	print Theta
#
#
# == Function 2j. ===
#
# This function plots decisions boundaries in addition to the plotting of the 
# given training set.
#
# Arguments: None
#			
# Local variables: data -- Training set to be plotted
#				   Y -- Y component of training set
#				   X -- Array storing the monomials obtained from the X's training
#						set components
#				   Theta -- Parameters of the logistic regression
#				   Lambdas -- Array of all lambda values to be considered for the
#							 regularized regression
#				   Lambda -- Real variable looping over all values of Lambdas
#				   jcost -- value of cost function for a given Theta and Lambda
#				   xGrid -- Array of grid points along the X-axis
#				   yGrid -- Array of grid points along the Y-axis
#				   Z -- Array storing the points of the decision boundary 
#				   i, j -- Just counters
#				 
#				   
# IMPORTANT NOTE. This function only prints out the array and monomials and its calling
# 				  does not affect the general purpouse of this program.
#					
#
def PlotBoundaries():
	# We read training set from file...
	data  = genfromtxt(EX_DIRECTORY_PATH + "/Ex2_data_02.txt", delimiter = ',' )
	# From the data we extract off the Y component of them
	Y 	  = data[:,2]
	# From the X's components of the training set we form the monomials
	# to be the features of the regression
	X 	  = MonomialsFeatures( data[:, 0], data[:, 1] )
	# We set the theta parameter to be initially zero
	Theta = zeros( shape(X)[1] )
	#lamdas = [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100]
	# We set the array of Lambdas will all the desired values of lambda
	# to be considered
	Lambdas = [0.001, 1.0, 10.0]
	# We loop over all the selected values of lambdas.
	# For each value of lambda we...
	for Lambda in Lambdas:
		# ... determine the minimizing Theta and the minimized
		# cost function
		print "==============================================="
		print "		"
		print "Minimizing for lambda = " , Lambda
		print "		"
		Theta, jcost = MinimizingTheta( Theta, X, Y, Lambda )
		# For the plot to be generated we set its title with the 
		# current value of lambda...
		pyplot.title( "Lambda = %f" % (Lambda) )
		# ... and plot the training set. 
		PlotPosNeg( data )
		# We will now add the decision boundary for the given value of lambda
		# To do so:
		# We generate one-dimensional grid along the X axis
		xGrid = linspace( -1, 1.5, 50 )
		# and another one-dimensional grid along the Y axis
		yGrid = linspace( -1, 1.5, 50 )
		# We set an array of zeros with the length of the sum
		# of the recently generated one-dimensional grids
		Z = zeros( (len(xGrid), len(yGrid)) )
		# For each index in the x-grid...
		for i in range(0, len(xGrid)): 
			# and each index in the y-grid...
			for j in range(0, len(yGrid)):
				# We generate the monomials obtained from the XY grid,
				# store it in MappedAux
				MappedAux = MonomialsFeatures( array([xGrid[i]]), array([yGrid[j]]) )
				# and do the inner product with the minimizing Theta
				# to be stored in Z.
				Z[i,j] = MappedAux.dot( Theta )
		# We finally transpose it,
		Z = Z.transpose()
		# make a mesh from the X- and Y-grids
		xGrid, yGrid = meshgrid( xGrid, yGrid )
		# and plot the resulting decision boundary together with the 
		# training set	
		pyplot.contour( xGrid, yGrid, Z, [0.0, 0.0], label='Decision Boundary' )		
		pyplot.show()



def main():
	# We set the print options provided by SciPy
	# http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
	set_printoptions(precision=6, linewidth=200)
	# We first plot the training set
	print "Plotting training set..."
	PlotTrainingSet()
	#quit()
	# The following lines permit printing the array of monomials
	# needed for the regularized logistic regression and they can be
	# avoided.
	#print "Printing array of monomials..."
	#PrintMonomialsArray()
	#quit()
	Lambda = 0.0001
	print "Computing cost function... " 
	ComputeCostFunction(Lambda)
	#quit()
	print "Plotting decision boundary for a set of lambda values..."
	PlotBoundaries()
	
if __name__ == '__main__':
	main()