import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)


#Measure the accuracy of a model
#Train the model means create the model.
#Test the model means test the accuracy of the model.


#start with the data set we will test, 100 customers in a shop
x = numpy.random.normal(3, 1, 100)		#x = minutes before making purchase
y = numpy.random.normal(150, 40, 100) / x	#y = money spent on the purchase


#Display the OG data
plt.scatter(x, y)
plt.show() 


#Split into train and test
train_x = x[:80]	#Training set should be a random set of 80% of the OG data
train_y = y[:80]

test_x = x[80:]		#Testing set should be the remaining 20%
test_y = y[80:] 


#Display the Training set
plt.scatter(train_x, train_y)
plt.show() 


#Display the Testing set
plt.scatter(test_x, test_y)
plt.show() 


#For this set it looks like the best course of action to fit the data set is polynomial regression, so let's do that.
#Let's fit a line to the training data using poly reg
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show() 


#R2/R-squared value from 0 to 1 for how well the Training data fits in a polynomial regression
r2 = r2_score(train_y, mymodel(train_x))
print(r2) 

#Same for Testing data
r2 = r2_score(test_y, mymodel(test_x))
print(r2)

#Predicting a future value, in this case how much money will a customer spend if they stay in the shop for 5 minutes?
print(mymodel(5)) 