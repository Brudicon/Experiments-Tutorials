from random import randint
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

########################Generate Training Set#######################
TRAIN_SET_LIMIT = 1000
TRAIN_SET_COUNT = 100

TRAIN_INPUT = list()
TRAIN_OUTPUT = list()
for i in range(TRAIN_SET_COUNT):
    a = randint(0, TRAIN_SET_LIMIT)	#Technically a form of multiple regression (linear but with multiple x-values)
    b = randint(0, TRAIN_SET_LIMIT)
    c = randint(0, TRAIN_SET_LIMIT)
    op = a + (5*b) + (3*c)		#Can't imagine this code method is useful because it requires you to know the pattern of the data already
    TRAIN_INPUT.append([a, b, c])
    TRAIN_OUTPUT.append(op)

################Create and Train LinearRegression model #############
predictor = LinearRegression(n_jobs=-1)
predictor.fit(X = TRAIN_INPUT, y = TRAIN_OUTPUT)

#############################Testing Data############################
X_TEST = [[10, 20, 30]]
outcome = predictor.predict(X=X_TEST)
coefficients = predictor.coef_

print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))