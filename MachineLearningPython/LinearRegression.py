import matplotlib.pyplot as plt
from scipy import stats


#Input Data
#x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
#y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]


#return important key values of linear regression
slope, intercept, r, p, std_err = stats.linregress(x, y)


#func to calculate slope intercept form of the data
def myfunc(x):
  return slope * x + intercept


#run each value of the x-array through myfunc, result is new array for y-axis
mymodel = list(map(myfunc, x))


#draw scatter plot of original data
plt.scatter(x, y)


#Draw line of linear regression (slope line of the data)
plt.plot(x, mymodel)


#Relationship (coefficient of correlation, how well it fits the data from 0 to +-1)
print('\nRelationship: {}'.format(r))
if -.5 <= r or r <= .5:
	print('Linear Regression will likely not yield accurate predictions for this data set. Highly recommended that you use a more accurate method for this spread-out of a dataset.\n') 

#Predict a value based on the slope intercept form of the data set
predX = 10
print('If x = {}, then y = {} \n'.format(predX, myfunc(predX)))

#display the diagram
plt.show() 

