#
# SECTION I. Code information 
#
# This program contains a set of functions corresponding to a
# RECOMMENDER SYSTEM
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
# This program executes the collaborative filtering learning algorithm used for
# a recommender system and applied to a dataset of movie ratings. The dataset is
# obtained from the MovieLens 100k Dataset from GroupLens Research
# http://grouplens.org/datasets/movielens/
# The code makes use of the following files:
#
# movies.mat -- Movie Review Dataset
#				This file stores two matrices:
#
#				  - Y (double).- matrix of ratings (1682 x 943)
#					The elements of this matrix, Y_ij
#					provide the rating of movie i given by user j
#					(if the raing is provided)
#
#				  - R (logical).- matrix providing information IF users gave rating
#					to movies.
#					The elements of this matrix, R_ij are 1 if
#					movie i was rated by user j and zero otherwise.
#					
#
# movieParams.mat -- Parameters provided for debugging
#					 This file stores the following information:
#			
#					  - num_movies -- Number of movies in the dataset (1682)
#					  - num_users -- Number of users that rated movies (943)
#					  - num_features -- Number of features considered for movies (10)
#					  - Matrix Theta (double) -- Matrix of parameters for all users (943 X 10)
#							This matrix can be seen as 943 columns (one column per user)
#					  - Matrix X (double) -- Matrix of features for all movies (1682 X 10)
#							This matrix can be seen as 1682 columns (one column per movie)
#
# movie_ids.txt -- List of movies. This file contains information about all movies.
# 					The format is:
# 									idMovie | Movie title | (year)
#  									Example:
#   										11 Seven (Se7en) (1995)
#					The file contains information of 1682 movies (1682 lines)
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
#     2a. LoadListOfMovies() --> Loads list of movies from movie_ids.txt file
#
#     2b. NormalizeRatings( Y, R ) --> Normalize matrix of ratings Y
#
#	  2c. GetParameters( num_users, num_movies, num_features, params ) --> Obtains matrix
#						of features X and matrix of users' parameters theta
#
#	  2d. JCostFunct( params, Y, R, num_users, num_movies, num_features, Lambda ) -->
#					Computes the value of the cost function
#
#     2e. GradJCostFunct( params, Y, R, num_users, num_movies, num_features, Lambda ) -->
#					Computes the gradient of the cost function
#
#	  2f. PlotRatingMatrix() --> Plots the matrix of ratings
#
#	  2g. Rating_Predicting() --> 
#              
#
########################################################################################################################
#
#
#  SECTION V. Actual code
#
#
# We import the standard python modules through sys
#
import sys
# 
#
# Library for numerical calculations...
# http://www.numpy.org
from numpy import *
#
# Library for scientific calculations...
# http://www.scipy.org
import scipy.io, scipy.misc, scipy.optimize
#
# Library for plotting...
# http://matplotlib.org
from matplotlib import pyplot, cm, colors, lines
#
# http://matplotlib.org/basemap/users/installing.html
from mpl_toolkits.mplot3d import Axes3D
#
#
# We set the path of the code and files to be read for its execition
#
EX_DIRECTORY_PATH = '/Users/rogeliocuevassaavedra/Documents/mlclass/'
#
# 
# ==== Function 2a. ====
#
# This function reads the "movie_ids.txt" file that contains information
# about movies.
# The format of movies is:
# 
#  idMovie | Movie title | (year)
#  Example:
#   		11 Seven (Se7en) (1995)
#
# The file contains information of 1682 movies (1682 lines)
#
#
# Arguments: none
#
# Local variables: counter -- counter (surprising, right?)
#				   infoMoviesIds -- variable where the file info is loadad to
#				   contents -- variable containing the lines of infoMoviesIds
#				   content -- auxiliary variable to scan through contents
# 
#
# Outputs: movies -- a dictionary containing info on the movies
#
#
#
def LoadListOfMovies():
	# 
	# Set an empty dictionary (curly brackets indicate dictionary)
	#
	movies = {}
	#
	# Initialize counter
	# 
	counter = 0
	#
	# Open file and read its content
	# Note: 'rb' indicates read file in binary mode
	#
	with open(EX_DIRECTORY_PATH + "/movie_ids.txt", 'rb') as infoMovieIds:
		#
		# ... get its content line by line
		# Note: .readlines(), as suggested, indicates reading lines of file
		#
		contents = infoMovieIds.readlines()
		#
		# For every line in contents ...
		#
		for content in contents:
			#
			# ... add title of movie to array by striping and spliting strings
			#
			# Notes: .strip() returns a copy of the string with leading and trailing 
			#				  characters removed.
			#		 .split() return a list of the words in the string
			#				  while .split(' ',1) indicates affecting only the first (1) 
			#				  part of the string
			#        Recall the first index of an array is 0 (id of the movie)
			#        which is why we need index [1] (movie title along with year)
			#
			movies[counter] = content.strip().split(' ', 1)[1]
			#
			# ... increase counter in a unit. 
			#
			counter += 1
	#
	# return all movie titles
	#
	return movies
#
# 
# ==== Function 2b. ====
#
# This function normalizes the Y matrix so that its mean is zero
# and its standard deviation is one. The Y matrix is a rectangular 
# matrix of order (number of movies) X (number of users).
# We refer to this matrix as the matrix of ratings
#
# Arguments: Y -- Matrix of ratings
#            R -- Matrix indicating whether movies have been rated
#
# Local variables: i -- Counter
#				   idx -- Integer labeling where there is a rating for the movie
#				   num_movies -- Number of movies 
#
# Outputs: Y_mean -- Mean of matrix Y 
#		   Y_norm -- Matrix Y normalized
#
def NormalizeRatings( Y, R ):
	#
	# Get number of movies from matrix Y (number of rows in Y)
	#
	num_movies = shape( Y )[0]
	#
	# Inizialize Y_mean to a one-columned array of zeroes with number of rows
	# equal to number of movies (num_movies)
	#
	Y_mean = zeros((num_movies, 1))
	#
	# Initialize Y_norm to a matrix of zeros having the same shape 
	# as the matrix Y
	#
	Y_norm = zeros( shape( Y ) )
	#
	# For every movie the in the list...
	#
	for i in range( 0, num_movies ):
		#
		# Identify where there is a rating for the movie 
		# and get the label idx
		#
		idx = where( R[i] == 1 )
		#
		# Get the average of rating for that movie from 
		# the Y matrix
		#
		Y_mean[i] = mean( Y[i, idx] )
		#
		# And finally normalize the matrix by substracting the just computed
		# mean Y_mean
		#
		Y_norm[i, idx] 	= Y[i, idx] - Y_mean[i]
	#
	# return the requested Y_norm and Y_mean
	#
	return Y_norm, Y_mean
#
#
# 
# ==== Function 2c. ====
#
# This function obtains the matrices of features (X) and parameters (theta)
# based on the number of users, movies and featues obtained from the .mat files.
# This function supports the functions related with the computation of 
# the cost function (JCostFunct), its gradient (GradJCostFunct) and the 
# recommender system itself (Rating_Predicting)
#
# Arguments: num_users -- Number of users in dataset
#            num_movies -- Number of movies in dataset
#			 num_features -- Number of features
#			 params -- Collective set of parameters 
#						(features X and users' parameters theta)
#
# Local variables: i -- Counter
#				   idx -- Integer labeling where there is a rating for the movie
#				   num_movies -- Number of movies 
#
# Outputs: X -- Matrix of features
#		   theta -- Matrix of users' parameteres
#
def GetParameters( num_users, num_movies, num_features, params ):
	#
	# From the collective set of parameters, get the section corresponging
	# to X (matrix of features)...
	#
	X  = params[:num_movies * num_features]
	#
	# and the section corresponding to theta (matrix of parameters)
	#
	theta 	= params[num_movies * num_features:]
	#
	# We now re-shape X as a rectangular matrix num_features X num_movies
	# and transpose it. 
	#
	# Note: .reshape and .transpose do exactly what they suggest
	#	    For further reference: 
	#
	X = X.reshape( (num_features, num_movies) ).transpose()
	#
	# We do something simiar with theta ...
	#
	theta 	= theta.reshape( num_features, num_users ).transpose()
	#
	# ... and return them as outputs of the function
	#
	return X, theta


#
# 
# ==== Function 2d. ====
#
# This function computes the cost function associated with matrices of features (X), 
# parameters (theta) and ratings (Y)
#
# Arguments: params -- Collective matrix of matrix of features X and 
#					   matrix of parameters theta
#            Y -- Matrix of ratings
#			 R -- Boolean matrix (1 or 0) indicating what movies have been reated (1)
#				  and which ones have not (0)
#			 num_users -- Number of users
#			 num_movies -- Number of movies
#			 num_features --  Number of features 
#			 Lambda -- Regularization parameter
#
# Local variables: X -- Matrix of features
#				   theta -- Matrix of users' parameters
#				   regularization -- Value of regularization
#
# Outputs: J -- Cost function
#		   
#
def JCostFunct( params, Y, R, num_users, num_movies, num_features, Lambda ):
	# 
	# Using the GetParameters function (see above) we obtain
	# the ***** matrices
	#
	X, theta = GetParameters( num_users, num_movies, num_features, params )
	#
	# We compute the cost function without regularization; 
	# square of the difference between the
	# actual rating matrix and the estimated one obatined from X and theta...
	#
	J = 0.5 * sum( (X.dot( theta.T ) * R - Y) ** 2 )
	#
	# ... and compute the regularization term associated with theta, X and
	# the regularization parameter Lambda
	#
	regularization = 0.5 * Lambda * (sum( theta**2 ) + sum(X**2))
	#
	# We update the cost function by adding the regularization value
	#
	J = J + regularization
	#
	# Return value of cost function
	#
	return J 
#
#
# 
# ==== Function 2e. ====
#
# This function computes the gradient of the cost function associated with ....
#
# Arguments: params --
#            Y -- Matrix of ratings
#			 R -- Boolean matrix indicated whether the movies are ranked
#			 num_users -- Number of users
#			 num_movies -- Number of movies
#			 num_features --  Number of features 
#			 Lambda -- Regularization parameter
#
# Local variables: X -- Matrix of features
#				   theta -- Matrix of parameters
#				   error_Matrix -- Matrix difference between estimated
#								   matrix of rating and actual one
#				   X_Grad -- Gradient of features
#				   theta_Grad -- Gradient of parameters
#
# Outputs: Concatenated transpose of X_Grad and theta_Grad
#		   
#
def GradJCostFunct( params, Y, R, num_users, num_movies, num_features, Lambda ):
	#
	# Extract off the X and theta matrices by means of the GetParameters function
	# (see function above)
	#
	X, theta 	 = GetParameters( num_users, num_movies, num_features, params )
	#
	# Compute the error matrix as the difference between the estimated rating matrix
	# associated with theta and X and the actual rating matrix
	#
	error_Matrix = X.dot( theta.T ) * R - Y
	#
	# We compute gradient of parameters and features, respectively
	#
	X_Grad 		 = error_Matrix.dot( theta ) + Lambda * X
	theta_Grad 	 = error_Matrix.T.dot( X ) + Lambda * theta
	#
	# We concatenate the X_Grad and theta_Grad matrices by rows (r_)
	# after flattening them
	#
	return r_[X_Grad.T.flatten(), theta_Grad.T.flatten()]

#
#
# ==== Function 2f. ====
#
# This function plots the rating matrix 
#
# Arguments: none
#
# Local variables: Y_R_matrices -- Matrix where Y and R will be collectivly stored
#				   Y -- Matrix of ratings
#				   R -- Logical matrix indicating whether movies have been rated
#				   
# Outputs: Matrix of ratings displayed on screen
#		   
#
#
def PlotRatingMatrix():
	#
	# Load the Movie Review Dataset from movies.mat file
	# Remember this files stores two matrices:
	#
	#			  - Y (double).- matrix of ratings (1682 x 943)
	#					The elements of this matrix, Y_ij
	#					provide the rating of movie i given by user j
	#					(if the raing is provided)
	#
	#			   - R (logical).- matrix providing information IF users gave rating
	#					to movies.
	#					The elements of this matrix, R_ij are 1 if
	#					movie i was rated by user j and zero otherwise.
	#					
	#
	# We load the matrices from the MATLAB/Octave file "movies.mat"
	# by means of the SciPy library
	# http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.io.loadmat.html
	#
	Y_R_matrices = scipy.io.loadmat(EX_DIRECTORY_PATH + "/movies.mat") 
	#
	# Exctract off the two matrices...
	#
	Y, R = Y_R_matrices['Y'], Y_R_matrices['R']

	#print mean( extract ( Y[0,:] * R[0,:] > 0, Y[0, :] ) )

	# We use .imshow, which is part of the 2D plotting library "matplotlib",
	# in order to have a graphical representation of the array Y.
	# For further reference: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow
	# 
	pyplot.imshow( Y )
	#
	# ...and add the the plot X and Y labels correspoding to users and movies, respectively
	#
	pyplot.xlabel( 'Users')
	pyplot.ylabel( 'Movies' )
	#
	# ...and display the figure using .show()
	# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.show
	#
	pyplot.show()
#
#
#
# ==== Function 2g. ====
#
# This function can be considered as the actual "recommender system." It requests
# the user to rank a set of 10 movies and predicts ratings for other 10 movies.
#
# Arguments: none
#
# Local variables: ids_to_rate -- Array of integers labelling the movies that will be ranked
#								  by the user.
#				   movies -- Dictionary where information (ids and names) will be loaded to.
#				   user_ratings -- Column vector where the ratings of user will be stored.
#				   Y_R_matrices -- Matrix where Y and R will be collectivly stored
#				   Y -- Matrix of ratings
#				   R -- Logical matrix indicating whether movies have been rated
#				   i, j -- integers utilized for loops 
#   			   Y_norm -- Normalized Y matrix
#   			   Y_mean -- Mean of Y matrix
#   			   num_movies -- number of movies
#   			   num_features -- number of features
#   			   X -- Matrix of features 
#   			   theta -- Matrix of users' parameters
#				   initial_params -- Initial guess of parameters for the recommender system 
#					  				(randomly generated )
# 				   Lambda -- Regularization parameter
#				   min_result -- Stores results of minimization (cost function and parameters
#								 of the minimization). Minimization is performed nonlinear
#								 conjugate gradient
#				   final_params -- Final parameters of the iteration process of the 
#									 recommender system
#				   Jcost -- Value of the cost function
#				   prediction_preMean -- Results of the predicition prior accounting for their
#										 mean
#				   prediction -- Actual predictions (mean accounted for)
#				   idx -- Array of integeres corresponding to the sorter predictions
#				
#
#
# Outputs: Predicted ratings displayed on screen
#		   
#
def Rating_Predicting():
	
	print "Now you will be asked to rate 10 movies,"
	print "please provide a number between 1 (worst) to 5 (best)"
	#
	# We set in an array the ids of the movies that will be rated
	#
	ids_to_rate = [0,97,6,11,53,63,65,68,182,225,354]
	#	
	# We load the list of movies
	#
	movies = LoadListOfMovies()
	#
	# Set initially the ratings to a vector column of zeroes
	#
	user_ratings = zeros((1682, 1))
	#
	#
	# We request the user to rate 10 pre-selected movies...
	#
	print "				"
	print "				"
	for i in range(0,10):
	#	rand_id = randint(0,1682)
		print "Your rating for the movie " , movies[ids_to_rate[i]]
		user_ratings[ids_to_rate[i]] = input("is?...")
	
	#
	# ...and show to the user, on screen, the ratings he just provided 
	#
	print "			  "
	print "You have rated: "
	for i in range( 0, 1682 ):
		#
		# We make sure the movies have actually been rated
		# by means of user_ratings[i] and use the formats
		# %d (decimal) and %s (string) for printing
		# See https://docs.python.org/2/library/stdtypes.html
		# for further information
		# 
		#
		if user_ratings[i] > 0:
			print " %d for %s" % (user_ratings[i], movies[i])
	#
	print "					"
	print " Please wait, predictions being computed..."

	#
	# We load the matrix of ratings (Y) and matrix indicating whether 
	# movies have been rated (R) from .mat file...
	#
	Y_R_matrices = scipy.io.loadmat(EX_DIRECTORY_PATH + "/movies.mat") 
	#
	# ... and extract off the aformentioned matrices
	#
	Y, R = Y_R_matrices['Y'], Y_R_matrices['R']

	#
	# We concatenate the user_ratings array and the matrix of ratings (Y)
	# by columns (c_) and store it back to Y
	#
	Y = c_[user_ratings, Y]
	#
	# We concatenate the user_ratings (provided they are greter than zero) array 
	# and the matrix R (matrix that indicates whether the movies have been rated)
	# by columns (c_) and store it back to R
	#
	R = c_[user_ratings > 0, R]
	#
	# We normalize the Y and R matrices (see function NormalizeRatings)
	#
	Y_norm, Y_mean = NormalizeRatings( Y, R )
	#
	# We get number of users and movies from rows and columns of Y, respectively
	#
	num_movies, num_users = shape( Y )
	#
	# We set manually the number of features
	#
	num_features = 10
	#
	# We initialize the features (X) and parameters (theta) with random numbers
	#
	X 		= random.randn( num_movies, num_features )
	theta 	= random.randn( num_users, num_features )
	#
	# We concatenate the X and theta matrices by rows (r_)
	# after transposing and flattening them
	#
	initial_params = r_[X.T.flatten(), theta.T.flatten()]
	#
	# We manually set the regularization parameter (Lambda)
	#
	Lambda = 200.0
	# 
	#
	# We now proceed to minimize the cost function by means of 
	# nonlinear conjugate gradient: 
	#   https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
	#  
	# To do so, we use the scipy.optimize.fmin_cg library from SciPy
	# http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.fmin_cg.html
	#
	# We manually set the maximum number of iterations to 200 (maxiter = 200)
	# and decide no to display a convergence message (disp = False)
	# full_output is set to True as we decide to keep the value of the cost
	# function and parameters of the minimization to determine X and theta
	#
	#
	min_result = scipy.optimize.fmin_cg( JCostFunct, fprime=GradJCostFunct, x0=initial_params, \
									args=( Y, R, num_users, num_movies, num_features, Lambda ), \
									maxiter=200, disp=False, full_output=True )

	#
	# We collect the results of the optimization, min_result, and store them
	# in parameters and value of cost function respectively
	#
	final_params, Jcost = min_result[0], min_result[1]

	#
	# We use the recenty computed parameters to determine features and parameters
	# (X and theta) by means of GetParameters (see function)
	#
	X, theta = GetParameters( num_users, num_movies, num_features, final_params )
	#
	# We easly determine the prediction from a simple inner product of
	# parameters and features. Keep in mind this prediction is obtained 
	# from normalized data and the mean of Y....
	#
	prediction_preMean = X.dot( theta.T )
	#
	# ...which we now account for by adding back Y_mean
	#
	prediction = prediction_preMean[:, 0:1] + Y_mean
	#
	# We extract the indices that sort the prediction array
	# by means of the numpy.argsort library
	# http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
	# axis = 0 indicates sorting along the first axis.
	# This sorting indices are needed for...
	# 
	idx = prediction.argsort(axis=0)[::-1]
	#
	# ...getting the predictions in order
	#
	prediction = prediction[idx]
	#
	# We print for the user the result of the predited ratings....
	#
	print "					"
	print "Here are the predicted ratings:"
	print "					"
	#
	# ... and do so for the first 10 movies
	#
	for i in range(0, 10):
		j = idx[i, 0]
		#
		# As before, %s indicates string 
		# while %.1f indicates floating point rounded to one decimal
		#
		print "Predicting rating %.1f for movie %s" % (prediction[j], movies[j])
#
#
# Body of the main program
#
def main():

	#
	# We first plot the matrix of ratings...
	#
	print "Plotting rating matrix..."
	PlotRatingMatrix()
	#
	# ... and then put the actual recommender system to work
	#
	print "			  "
	print "Let's rate... "
	Rating_Predicting()


if __name__ == '__main__':
	main()