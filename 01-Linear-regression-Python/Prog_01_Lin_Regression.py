#
# SECTION I. Code information 
#
# This program contains a set of functions corresponding to 
# LINEAR REGRESSION
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
# This program reads data from the Ex1_data_01.txt file, performs a 
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
#     2a. Id_5() --> Prints out the 5X5 identity matrix to screen.
#                   Note. This function is not needed for linear regression but is 
#                         coded in order to make a compleate version of the 
#                         course's assignment No. 1
#
#     2b. Hypothesis(X, Theta) --> This function computes the hypothesis function for linear regression
#                                   
#     2c. CostFunct(X, y, Theta) --> This function computes the cost function (error function) for linear regression
#
#     2d. GradientDescent(X, Y, Theta, Alpha, NumIter) --> This function executes the gradient descent algorithm
#
#     2e. PlotXY(X, Y) --> This function plot a set of given ordered pairs (X,Y).
#
#     2f. PlotLearningSet() --> This function obtains a training set and plots it using the
#                               PlotXY(X,Y) function
#
#     2g. DataGradDesc() --> This function obtains a training set from a file and does gradient
#                            descent using such a training set by invoking the function
#                            GradientDescent(X, Y, Theta, Alpha, NumIter)
#
#     2h. PlotData_Surface_Countour(Xmin,Xmax,Npts_X,Ymin,Ymax,Npts_Y)
#                   ---> This function generates a surface (3D plot) and counter plot
#                        in the [Xmin,Xmax] X [Ymin,Ymax] 2D domain and the chosen
#                        number of points corresponding to each dimension
#      
#
########################################################################################################################
#
#  SECTION V. Actual code
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
# We set up the directory path where the data will be read from
# Such a directoy is stored in EX1_DIR_PATH
EX1_DIR_PATH = '/Users/rogeliocuevassaavedra/Documents/mlclass/'
#
#
# ==== Function 2a. ====
#
# This function prints out to screen the 5X5 identity matrix.
# Not needed for the linear regression
#
def Id_5():
    A = eye(5)
    print A
#
#
# ================================================================
#
#
#
# ==== Function 2b. ====
#
# This vectorized function computes the vectorized hypothesis function for linear regression
# 
# Arguments: X -- Argument of hypothesis function
#            Theta -- Parameter of hypothesis function
#
def Hypothesis(X, Theta):
#
#   The functions is easily vectorized through the dot product...
#
    return X.dot(Theta)
#
# ================================================================
#
#
# ==== Function 2c. ====
#
# This vectorized function computes the cost (error) function for linear regression
# 
# Arguments: X -- Argument of hypothesis function
#            Y -- Y values of the training set 
#            Theta -- Parameter of hypothesis function
#
# Local variables: m -- Number of data
#                  AuxDif -- Error in values of Y 
#                            (computed from hypothesis minus Y coordiante of learning set)

def CostFunct(X, Y, Theta):
    #
    ## We first determine the number of data; lenght of Y
    ## and store it in m
    #
    m 	 = len(Y)
    #
    # We compute the error (difference between lerning Y set and computed one)
    # and store it an AuxDif...
    AuxDif = Hypothesis(X, Theta) - Y
    #
    # ...so that we can compute the cost function as a dot product (vectorized form)
    #
    return (AuxDif.T.dot(AuxDif) / (2 * m))[0, 0]
#
#
#
# ================================================================
#
#
# ==== Function 2d. ====
#
# This vectorized function executes gradient descent for 
# a given value of alpha and certain number of iterations 
# and returns the value of the gradient
# 
# Arguments: X -- Argument of hypothesis function
#            Y -- Y values of the training set 
#            Theta -- Parameter of hypothesis function
#            Alpha -- Given coefficient for gradient descent
#            NumIter -- Given number of iterations
#
# Local variables: m -- Number of data
#                  AuxDif -- Error in values of Y 
#                  Grad -- Value of gradient
#                  counter -- Counter to loop over number of iterations
#                  Inner_sum -- Stores cost (error) term in the iteration
# 
#
def GradientDescent(X, Y, Theta, Alpha, NumIter):
    ## We store the number of data in m...
    m    = len(Y)
    ## We copy Theta to gradient
    Grad = copy(Theta)
    ##
    ## For every iteration...
    ##
    for counter in range(0, NumIter):
        ## Compute cost (error) and store it in Inner_sum...
        Inner_sum = X.T.dot(Hypothesis(X, Grad) - Y)
        ## and update the corresonding gradient term
        Grad 	 -= Alpha * Inner_sum / m
    ##
    ## Finally get gradient 
    ##
    return Grad
#
#
# ================================================================
#
#
# ==== Function 2e. ====
#
# This function creates a X vs. Y plot based on the (X,Y) learning set.
# Notes: 1. X should exclude the intercept units.
# 
# Arguments: X -- X values of training set
#            Y -- Y values of training set 
#
# Local variables: NO local variables
# 
#
#
def PlotXY(X, Y):
#    Call pyplot.show(block=True) in order to show the plot window
#   Generate plot with the given X and Y data with selected markers
    pyplot.plot(X, Y, 'rx', markersize=6.5 )
#   Add X and Y labels
    pyplot.xlabel('City Population in 10,000s')
    pyplot.ylabel('Profit in $10,000s')
#
# ================================================================
#
#
# ==== Function 2f. ====
#
# This function reads (X,Y) data from "Ex1_data_01.txt" file
# and plots them by calling out the function PlotXY(X, Y)
# 
# Arguments: NO explicit arguments; arguments are obtained through the reading of the data file
#
# Local variables: data - set of data read from file
#                  X, Y - Ordered pairs of selected data
#                  m - Number of data in file
#
def PlotLearningSet():
    # Read data from txt file and store them in "data"
    data = genfromtxt( EX1_DIR_PATH + "Ex1_data_01.txt", delimiter=',')
    # Separate data into X and Y
    X, Y = data[:, 0], data[:, 1]
    # Determine number of data 
    m 	 = len(Y)
    #Y 	 = Y.reshape(m, 1)
    # We plot training set
    PlotXY(X, Y)
    # Call pyplot.show(block=True) in order to show the plot window
    pyplot.show(block=True)
#
#
# ================================================================
#
#
# ==== Function 2g. ====
#
# This function reads (X,Y) data from "Ex1_data_01.txt" file
# and applies gradient descent to them by calling out the function
# GradientDescent(X, Y, Theta, Alpha, NumIter)
# 
# Arguments: NumIter - Number of iterations
#            Alpha - Alpha parameter for gradient descent
#
# Local variables: data - set of data read from file
#                  X, Y - Ordered pairs of selected data
#                  m - Number of data in file
#                  Theta - Optimizing parameter to be determined
#                  cost - value of cost function
#
#
def DataGradDesc(NumIter,Alpha):
    # Read data from 'Ex1_data_01.txt' file
    data = genfromtxt( EX1_DIR_PATH + 'Ex1_data_01.txt', delimiter=',')
    # Determine X and Y values from data
    X, Y = data[:, 0], data[:, 1]
    # Obtain number of data
    m 	 = len(Y)
    # Reshape Y values as a column vector m X 1
    Y 	 = Y.reshape(m, 1)
    # Add a column of ones to the X matrix
    X 			= c_[ones((m, 1)), X]
    # Set the initial values of theta to zero
    Theta 		= zeros((2, 1))
    # We print the number of iterations...
    print 'Number of iterations ' , NumIter
    # ... and value of the alpha parameter 
    print 'Value of alpha ' , Alpha
    # We compute the initial value of the cost function
    cost 	= CostFunct(X, Y, Theta) 
    print 'Initial value of cost function ' , cost 
    # We determine the minimizing value of Theta by invoking the
    # gradient descent algorithm
    Theta 	= GradientDescent(X, Y, Theta, Alpha, NumIter)
    # We compute the final value of the cost function
    cost    = CostFunct(X, Y, Theta)  
    print 'Final value of cost function ' , cost 
    # And print the minimizing Theta
    print 'Theta transpose = ' , Theta.reshape(1,2)
    # We plot again the original training set
    print 'Plotting the original training set...'
    PlotXY(X[:, 1], Y)
    # and include the equation resulting from the linear regression
    print '...along with the equation resulting from the linear regression'
    pyplot.plot(X[:, 1], X.dot(Theta), 'b-')
    # Call pyplot.show(block=True) in order to show the plot window
    pyplot.show(block=True)

#
# ================================================================
#
#
# ==== Function 2h. ====
#
# This function reads (X,Y) data from "Ex1_data_01.txt" file
# and generates surface and contour plot from them.
# 
# Arguments: Xmin - Minimum X value for the points generated along the X axis
#            Xmax - Maximum X value for the points generated along the X axis
#            Npts_X - Number of points to be generated along the X axis
#            Ymin - Minimum X value for the points generated along the X axis
#            Ymax - Minimum X value for the points generated along the X axis
#            Npts_Y - Number of points to be generated along the Y axis

#
# Local variables: data - set of data read from file
#                  X, Y - Ordered pairs of selected data
#                  m - Number of data in file
#                  ThetaX_vals, ThetaY_vals - Arrays containing equally spaced numbers 
#                                             in the X and Y direction.
#                                             These are used to generate a grid of points.
#                  Jvals - Matrix storing the value of the cost function evaluated in the
#                          grid generated by ThetaX_vals & ThetaY_vals
#                  
#
#
def PlotData_Surface_Countour(Xmin,Xmax,Npts_X,Ymin,Ymax,Npts_Y):
    # Read data from 'Ex1_data_01.txt' file...
    data = genfromtxt( EX1_DIR_PATH + "/Ex1_data_01.txt", delimiter=',')
    # Select X and Y data from the read data
    X, Y = data[:, 0], data[:, 1]
    # Determine number of data
    m 	 = len(Y)
    # Re-shape Y values to make them a vector
    Y 	 = Y.reshape(m, 1)
    # Add a column of ones to X
    X 	 = c_[ones((m, 1)), X]
    # We generate evenly spaced Nptx_X numbers along the X and Npts_Y along the Y axis 
    # in the corresponding chosen intervals
    ThetaX_vals = linspace(Xmin,Xmax,Npts_X)
    ThetaY_vals = linspace(Ymin, Ymax,Npts_Y)

    # We initialize the cost function matrix with zeroes
    Jvals = zeros((len(ThetaX_vals), len(ThetaY_vals)), dtype=float64)
    # From every index in the X grid...
    for i, vX in enumerate(ThetaX_vals):
        # ... and every index in the Y grid...
        for j, vY in enumerate(ThetaY_vals):
            # ...we generate and order pair of X,Y values and store it in the
            # Theta array as a 2 X 1 vector...
            Theta 		= array((ThetaX_vals[i], ThetaY_vals[j])).reshape(2, 1)
            # ...from which we compute the cost function associated with each ordered pair
            # index (i,j)
            Jvals[i, j] = CostFunct(X, Y, Theta)

    # We now generate a mesh of (X,Y) points
    R_X, R_Y = meshgrid(ThetaX_vals, ThetaY_vals)

    # We invoke "figure" to manage the plot elements displayed in the PLOT SURFACE
    fig = pyplot.figure()
    # We generate the X axis
    ax 	= fig.gca(projection='3d')
    # and finally the plot surface
    ax.plot_surface(R_X, R_Y, Jvals)
    pyplot.show(block=True)

    # We do something similar fot the contour plot...
    # We invoke "figure" to manage the plot elements displayed in the CONTOUR PLOT
    fig = pyplot.figure()
    # And generate the contour plot obtained from a logarithmically spaced integers
    pyplot.contourf(R_X, R_Y, Jvals.T, logspace(-2, 3, 20))
    pyplot.plot(Theta[0], Theta[1], 'rx', markersize = 30)
    pyplot.show(block=True)


def main():
    set_printoptions(precision=6, linewidth=200)

#
# We print out the 5X5 identity matrix...
    print '     '
    print 'The 5X5 identity matrix is given by:'
    print '     '
    Id_5()
    print '     '
    print 'Plotting the learning set...'
    PlotLearningSet()
    print 'Running gradient descent...'
    DataGradDesc(2000,0.01)
    print 'Plotting surface and contour plot...'
    PlotData_Surface_Countour(-10,10,100,-4,4,100)
    print 'DONE'
    sys.exit()


if __name__ == '__main__':
    main()
