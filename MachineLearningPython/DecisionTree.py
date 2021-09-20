import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg


#Decision Trees are flow charts based on previous experience.
#May return different values over multiple executions because it is not 100% certain on the answer.
#Based on the probability of the outcome.


#read the dataset into a dataframe#################################################
df = pandas.read_csv("shows.csv")
print(df) 

#Convert string values into numerical values#######################################
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
print(df) 

#separate the "feature" columns (X) from the "target" column (Y)###################
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']
print(X)
print(y)


#Create the actual decision tree, fit it with our details, and save as a png#######
dtree = DecisionTreeClassifier()

dtree = dtree.fit(X, y)

data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)


#"Should I go see a show with a 40 yr old american comedian with 10 yrs of experience and rank 7?"
print(dtree.predict([[40, 10, 7, 1]])) 
print('[1] means YES')
print('[2] means NO')


#Display the tree
plt.show() 



#There are many ways to split the samples, we use the GINI method in this tutorial.
#The Gini method uses this formula:
#Gini = 1 - (x/n)2 - (y/n)2
#Where x is the number of positive answers("GO"), n is the number of samples, and y is the number of negative answers ("NO"), which gives us this calculation:
#1 - (7 / 13)2 - (6 / 13)2 = 0.497