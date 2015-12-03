#
# SECTION I. Code information 
#
# This program contains a set of functions corresponding to 
# SUPPORT VECTOR MACHINE (SVM)
# 
#
# Coded by Rogelio Cuevas-Saavedra (rogercuevas.ca@gmail.com)
#
#
###########################################################################################################################
#
# SECTION II. General purpose of the program
#
# This program reads data from the Ex2_data_02.txt file, performs a 
# SUPPORT VECTOR MACHING (SVM) and plots the given training set as well as the 
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

import sys
import scipy.misc, scipy.io, scipy.optimize
from sklearn import svm, grid_search
from numpy import *

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlaba

from util import Util


###
# We set the directory where the the file is located
EX_DIRECTORY_PATH = '/Users/rogeliocuevassaavedra/Documents/Insight/'

#
# == Function 2a. ===
#
# This function plots the data set
# 
# Arguments: data -- Data to be plotted
#			 
#
# Local variables: positive -- Subset of the data set classified as positive ("1") 
#				   negative -- Subset of the data set classified as negative ("0")

def plot(data):
	#
	# From the given data set we extract off the positive and negative cases
	#
	positives = data[data[:, 2] == 1]
	negatives = data[data[:, 2] == 0]
	#
	# Once extracted, we plot them with different colors and symbols
	#
	pyplot.plot( positives[:, 0], positives[:, 1], 'ro' )
	pyplot.plot( negatives[:, 0], negatives[:, 1], 'yo' )

#
# == Function 2b. ===
#
# This function computes the value of a gaussian kernel 
# 
# Arguments: x1 -- First argument in the gaussian kernel function
#			 x2 -- Second argument in the gaussian kernel function
#			 sigma -- Standard deviation of the gaussian kernel
#			 
#
# Local variables: None
#
def GaussKernel(x1, x2, sigma):
	return exp( -sum((x1 - x2) **2.0) / (2 * sigma**2.0) )
#
#
# == Function 2c. ===
#
# This function plots the boundary decision
# 
# Arguments: X -- Data set provided
#			 TrainedSVM -- Variable controlling the type of SVM executed
#						   (linear, rbf, etc)
#			 
#
# Local variables: kernel -- Kernel to be used during the SVM
#				   w --
#				   xp --
#				   yp --
#			 	   x1plot -- 
#				   x2plot --
#				   X1 --
#				   X2 -- 
#				   vals -- 
#				   this_X -- 
#
#
#
def PlotBoundary( X, TrainedSVM ):
	kernel = TrainedSVM.get_params()['kernel']
	#
	#
	if kernel == 'linear':
		w 	= TrainedSVM.dual_coef_.dot( TrainedSVM.support_vectors_ ).flatten()
		xp 	= linspace( min(X[:, 0]), max(X[:, 0]), 100 )
		yp 	= (-w[0] * xp + TrainedSVM.intercept_) / w[1]
		pyplot.plot( xp, yp, 'b-')
		

	elif kernel == 'rbf':
		x1plot = linspace( min(X[:, 0]), max(X[:, 0]), 100 )
		x2plot = linspace( min(X[:, 1]), max(X[:, 1]), 100 )
		
		X1, X2 = meshgrid( x1plot, x2plot )
		vals = zeros(shape(X1))
		
		for i in range(0, shape(X1)[1]):
			this_X = c_[ X1[:, i], X2[:, i] ]
			vals[:, i] = TrainedSVM.predict( this_X )
		
		pyplot.contour( X1, X2, vals, colors='blue' )


#
#
# == Function 2d. ===
#
# This function generates a linear boundary
# 
# Arguments: None 
#			 
#
# Local variables: mat -- 
#				   X --
#				   Y --
#				   
#
#
def LinearBoundary():
	mat = scipy.io.loadmat( EX_DIRECTORY_PATH + "/ex6data1.mat" )
	X, Y = mat['X'], mat['y']

	plot( c_[X, Y] )
	pyplot.show( block=True )

	# linear SVM with C = 0.1
	linear_svm = svm.SVC(C=0.01, kernel='linear')
	linear_svm.fit( X, Y.ravel() )

	plot( c_[X, Y] )
	PlotBoundary( X, linear_svm )
	pyplot.show( block=True )	

	# try with C = 10
	linear_svm.set_params( C=1 )	
	linear_svm.fit( X, Y.ravel() )

	plot( c_[X, Y] )
	PlotBoundary( X, linear_svm )
	pyplot.show( block=True )	



	# try with C = 1000
	linear_svm.set_params( C=10000 )	
	linear_svm.fit( X, Y.ravel() )

	plot( c_[X, Y] )
	PlotBoundary( X, linear_svm )
	pyplot.show( block=True )	

#
#
#
#
#
#
def RBFBoundary(Cset,SigmaSet):
	x1 = array([1, 2, 1])
	x2 = array([0, 4, -1])
	sigma = 2

	#print "Gaussian kernel: %f" % GaussKernel( x1, x2, sigma )

	mat = scipy.io.loadmat( EX_DIRECTORY_PATH + "/ex6data2.mat" )
	X, y = mat['X'], mat['y']

	plot( c_[X, y] )
	pyplot.show( block=True )

	#sigma = 0.01
	sigma = SigmaSet
	#rbf_svm = svm.SVC(C=1, kernel='rbf', gamma = 1.0 / sigma ) # gamma is actually inverse of sigma
	rbf_svm = svm.SVC(C=Cset, kernel='rbf', gamma = 1.0 / sigma ) # gamma is actually inverse of sigma
	rbf_svm.fit( X, y.ravel() )

	plot( c_[X, y] )
	PlotBoundary( X, rbf_svm )
	
	pyplot.show( block=True ) 
	


def main():
	set_printoptions(precision=6, linewidth=200)
	LinearBoundary()

	Cset = 0.1 # 1
	SigmaSet = 0.1 # 0.01
	RBFBoundary(Cset,SigmaSet)

	Cset = 10 # 1
	SigmaSet = 0.001 # 0.01
	RBFBoundary(Cset,SigmaSet)
	
	

if __name__ == '__main__':
	main()