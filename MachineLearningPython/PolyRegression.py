import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#Used for data that follows a trend, but not a linear one


#Data
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]


#make the polynomial regression model
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))	#3 is the degree of the fitting polynomial, poly1d for 1 dimensional polynomial class
						#1=linear, 2=quadratic, 3=cubic, ect., highest exponent of the eq.

#Make the line, (lowest x-value, highest x-value, number of samples to generate)
myline = numpy.linspace(1, 22, 100)


#plot the data points
plt.scatter(x, y)


#plot the fit line through the data
plt.plot(myline, mymodel(myline))


#Relationship, in the form of R-squared for polynomial instead of just R for linear regression, 0-1
print(r2_score(y, mymodel(x))) 


#Predict a value based on the fit
pred = 25
print('x = {}, y = {}'.format(pred, mymodel(pred)))


#Show the diagram
plt.show()