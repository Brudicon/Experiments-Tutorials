import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

#Used to scale two disparaging value types into more easily comparable values (for instance, Weight and Volume)

df = pandas.read_csv("cars.csv")

X = df[['Weight', 'Volume']]

scaledX = scale.fit_transform(X)

print(scaledX) 