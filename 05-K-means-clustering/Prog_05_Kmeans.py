#
# SECTION I. Code information 
#
# This program contains a set of functions corresponding to 
# CLUSTERING - K-MEANS 
# 
# Coded by Rogelio Cuevas-Saavedra (rogercuevas.ca@gmail.com)
#
#
###########################################################################################################################
#
# SECTION II. General purpose of the program
#
# This program reads data from the Ex2_data_02.txt file, performs a 
# CLUSTERING - K-MEANS and plots the given training set as well as the 
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
#
#
from numpy import *
import scipy.misc, scipy.io, scipy.optimize, scipy.cluster.vq

###
from matplotlib import pyplot, cm, colors
from mpl_toolkits.mplot3d import Axes3D

###
# We set the directory where the the file is located
EX_DIRECTORY_PATH = '/Users/rogeliocuevassaavedra/Documents/Insight/'

#
# == Function 2a. ===
#
# This function sets the initial centroids in each cluster
# 
# Arguments: X -- Data to be analyzed
#			 K -- Number of clusters
#
# Local variables: None
#
def InitialCentroids( X, K ):
	# We initialize the centroids by simply choosing a
	# random point in each group
	return random.permutation( X )[:K]
#
#
#
# == Function 2b. ===
#
# This function associates all points in data to a corresponding
# closest centroid based on Euclidian metric
# 
# Arguments: X -- Data to be clustered
#			 Xcentr -- Coordinates of the centroids
#
# Local variables: K -- Number of clusters
#				   m -- Number of data in training set
#				   indx -- Array storing the centroids indices 
#						   associated with the clusters
#				   i -- just a counter
#				   dist_Threshold --
#				   lowest_index
#				   k -- integer labelling centroids
#				   dist_Xi_Xk -- distance between ith data point
#								 and kth centroid
#
def FindIndexCentroids( X, Xcentr,threshold ):
	# We determine the number of clusters based on the centroids
	K 	= shape( Xcentr )[0]
	# We determine the number of data in the training set
	m   = shape( X )[0]
	# We initialize the array of indices to zeroes
	indx = zeros( (m, 1) )
	#
	# The following cycles execute the actual determination of the indices
	# for the centroids in the clusters
	#
	# For each point in the training set ...
	for i in range(0, m):
		# 
		# We set a cost thershold to associate points to centroids.
		#			   = 100
		dist_Threshold = threshold
		# We 
		lowest_index = 0
		#
		# For each cluster...
		#
		for k in range( 0, K ):
			# We compute the vector difference of the ith data point
			# with the kth centroid...
			dist_Xi_Xk = X[i] - Xcentr[k]
			#
			# and compute the distance between them by taking the inner
			# product with itself
			dist_Xi_Xk = dist_Xi_Xk.T.dot( dist_Xi_Xk )
			#
			#
			# In case the distance between ith point and kth centroid
			# is less than threshold...
			if dist_Xi_Xk < dist_Threshold:
				# we update the distance threshold with the
				# just computed distance between centroid and point and...
				dist_Threshold = dist_Xi_Xk
				# and updated the lowest index with label of kth centroid
				lowest_index = k
		#
		# We feed the ith component of the vector index with the 
		# up-to-date lowest index
		indx[i] = lowest_index
	#
	# We finally determine the final vector if indices by adding one
	# to all of them (Python indexing starts at zero)
	return indx + 1 
	
#	
#
# == Function 2c. ===
#
# This function sets the initial centroids in each cluster
# 
# Arguments: X -- Data to be clustered
#			 indx -- Array of indices
#			 K -- Number of clusters
#
# Local variables: m -- Number of rows in data set
#				   n -- Number of columns in data set
#				   Xcentroids -- Centroids
#				   data -- Appended array storing data set X and indices
#				   ik -- Counter that loops over clusters
#				   Xk-- Data obtained from X associated with the kth cluster
#				   Kcount -- Number of entries associated with the kth cluster Xk
#				   i -- Just a counter
#
def computeCentroids( X, indx, K ):
	# 
	# We determine rows and columns from X
	m, n = shape( X )	
	# We initialize centroids to zero
	Xcentroids = zeros((K, n))
	# We append the cluster index to array X
	data = c_[X, indx] 
	# For each cluster...
	for ik in range( 1, K+1 ):
		# Exctrat from X the data that belong to the current kth cluster
		Xk 			= data[data[:, n] == ik] 
		# Count the number of entries associated with cluster Xk
		Kcount 			= shape( Xk )[0]		
		# We determine the center of the cluster by finind the geomertric center
		# (average) of the points in that cluster.
		for i in range( 0, n ):
			Xcentroids[ik-1, i] = sum(Xk[:, i]) / Kcount
	# We finally return the centroids
	return Xcentroids

	
#
# == Function 2d. ===
#
# This function plots the data set and centroids of each cluster
# 
# Arguments: X -- Data to be clustered
#			 indx -- Array of indices
#			 Xcentroids -- Centroids of the clusters
#
# Local variables: data -- Appended array storing data set X and indices
#				   data1,data2,data3 -- Subset of data corresponding to each cluster

def PlotDataAndCentroids(X,indx,Xcentroids):
	# We append array of indices to data set X
	data = c_[X, indx]
	#
	# Extract data that falls in to cluster 1, 2, and 3 respectively, and plot them out
	# with different colors (blue, red, green)
	#
	data_1 = data[data[:, 2] == 1]
	pyplot.plot( data_1[:, 0], data_1[:, 1], 'bo', markersize=6 )

	data_2 = data[data[:, 2] == 2]
	pyplot.plot( data_2[:, 0], data_2[:, 1], 'ro', markersize=6 )

	data_3 = data[data[:, 2] == 3]
	pyplot.plot( data_3[:, 0], data_3[:, 1], 'go', markersize=6 )
	#
	# Plot as well the centroids with a different symbol and greater size
	# using the same corresponding colors.
	#
	pyplot.plot( Xcentroids[0, 0], Xcentroids[0, 1], 'b*', markersize=22 )
	pyplot.plot( Xcentroids[1, 0], Xcentroids[1, 1], 'r*', markersize=22 )
	pyplot.plot( Xcentroids[2, 0], Xcentroids[2, 1], 'g*', markersize=22 )

	pyplot.show( block=True )
#
#
# == Function 2e. ===
#
# This function runs the k-means algorithm itself
# 
# Arguments: X -- Data to be clustered
#			 initial_centroids -- Initial position of the centroids before to be
#								  updated during the k-means execution
#			 MaxIter -- Maximum number of iterarions 
#		     epsilon -- Convergence criterion for the centroids
#			 threshold -- Real number controling the first iteration in distance between
#						  centroids
#			 plot -- Logic variable controlling plotting fetures
#
# Local variables: K -- Number of clusters
#				   m -- Number of rows in data set
#				   centroids -- current centroids in each iteration
#				   indx -- array of indices
#  				   DistCentr -- Array storing distances between centroids in two 
#								consecutive iterations
#				   iteration -- Integer looping over the maximum number of iterations
#				   centroids_init -- 
#
def runkMeans( X, initial_centroids, MaxIter,epsilon, threshold, plot=False ):
	#
	# We get number of clusters from inicial centroids
	K 			= shape( initial_centroids )[0]
	# From the data set X we determine the number of rows
	m = shape(X)[1]
	# 
	# We set the controids to be the initial centroids provided to this function
	#
	centroids 	= copy( initial_centroids )
	#
	# We initialize the array of inices
	indx 		= None
	#
	# We initialize the array of distances between centroids
	DistCentr = zeros( (K, m) )
	
	print "Iteration and distances  between initial and final centroids = " 
	#
	# For each iteration...
	for iteration in range( 0, MaxIter ):
		# 
		# We update the indices with the current centroids
		indx 		= FindIndexCentroids( X, centroids, threshold )
		# We set the initial centroids (centroids prior to actual iteration)
		centroids_init = centroids
		# We update centroids with the recently updated indices
		centroids 	= computeCentroids( X, indx, K )
		# We compute the array of square of distance diferences between previous and 
		# up-to-date centroids
		DistCentr =  (centroids-centroids_init)**2
		# Print number of iteration and array of actual distances between centroids
		print  iteration+1 , DistCentr.sum(axis=1)
		# We check if distance is less than epsion...
		if DistCentr.sum(axis=1).any() < epsilon:
			# and in case is less we stop the iterations and report how many
			# iterations were needed to converge centroirds
			print "Position of centers converged in " , iteration+1 , "iterations"
			# And plot the final state of the k-mean procedure
			PlotDataAndCentroids(X,indx,centroids)
			break

		# In case we turned on the plotting settings, we show the plot at every 
		# iteration
		if plot is True:
			PlotDataAndCentroids(X,indx,centroids)	

	return centroids, indx


def TestFindCentroids(threshold):
	mat = scipy.io.loadmat( EX_DIRECTORY_PATH + "/ex7data2.mat" )
	X 	= mat['X']
	K 	= 3
	#
	initial_centroids = array([[3, 3], [6, 2], [8, 5]])

	print "													  "
	print "=================================================================="
	print " We test finding centroids for initial centroids chosen to be: "
	#
	# For every cluster...
	for i in range(0,K):
		print i+1, initial_centroids[i,:]
	#
	indx = FindIndexCentroids( X, initial_centroids,threshold )
	#
	
	centroids = computeCentroids( X, indx, K )
	print "Centroids..."
	for i in range(0,K):
		print i+1,indx[i],centroids[i,:]

	#print centroids
	# should be 
	# [[ 2.428301  3.157924]
	#  [ 5.813503  2.633656]
	#  [ 7.119387  3.616684]]




def Execute_kMeans(MaxIter,epsilon,threshold):
	mat = scipy.io.loadmat( EX_DIRECTORY_PATH + "/ex7data2.mat" )
	X 	= mat['X']
	K 	= 3


	#centroids = array([[1.2, 1.2], [2.2, 2.2], [3.2, 3.2]])

	centroids = array([[3, 3], [3.2, 3.2], [3.5, 3.5]])
	centroids = InitialCentroids(centroids,K)


	#centroids = InitialCentroids( X, K )
	
	runkMeans( X, centroids, MaxIter,epsilon,threshold, plot=True )




def main():
	set_printoptions(precision=6, linewidth=200)
	threshold = 100
	TestFindCentroids(threshold)
	MaxIter = 10
	epsilon = 1E-06
	Execute_kMeans(MaxIter,epsilon,threshold)
	#part1_3()
	

if __name__ == '__main__':
	main()