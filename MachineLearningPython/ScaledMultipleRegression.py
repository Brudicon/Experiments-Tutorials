import pandas
from sklearn import linear_model

#Multiple Regression is to perform linear regression on multiple different x-value columns


#convert excel sheet file into a data frame
df = pandas.read_csv("cars.csv")


#determines which columns of the sheet/dataframe to act as the X or y values
X = df[['Weight', 'Volume']]	#x1, x2, x3, ect.
y = df['CO2']			#the column set as the y-value


#Scale Weight and volume to more comparable values, optional but may be useful?
scaledX = scale.fit_transform(X)


#fit the data set as X and y by linear regression
regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2) 


#Print Coefficients for x1, x2, ect. of regr, which means that the given y-val changes by the coefficient of the x-val in question for each 1 unit of it that changes, for instance: newY = oldY + (x1increase * x1coefficient)
print(regr.coef_) 