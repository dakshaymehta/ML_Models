from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
 
#  this is to load the dataset
def load_dataset(full_path):
	# this is to load the dataset as a numpy array
	data = read_csv(full_path, header=None)
	# this is to retrieve numpy array
	data = data.values
	# split into input and output elements
	X, y = data[:, :-1], data[:, -1]
	# label encode the target variable to have the classes 0 and 1
	y = LabelEncoder().fit_transform(y)
	return X, y
 
# define the location of the dataset
full_path = 'glass.csv'
# load the dataset
X, y = load_dataset(full_path)
# define model to evaluate
model = RandomForestClassifier(n_estimators=1000)
# fit the model
model.fit(X, y)
# known class 0 (class=1 in the dataset)
row = [1.52101,13.64,4.49,1.10,71.78,0.06,8.75,0.00,0.00]
print('>Predicted=%d (expected 0)' % (model.predict([row])))
# known class 1 (class=2 in the dataset)
row = [1.51574,14.86,3.67,1.74,71.87,0.16,7.36,0.00,0.12]
print('>Predicted=%d (expected 1)' % (model.predict([row])))
# known class 2 (class=3 in the dataset)
row = [1.51769,13.65,3.66,1.11,72.77,0.11,8.60,0.00,0.00]
print('>Predicted=%d (expected 2)' % (model.predict([row])))
# known class 3 (class=5 in the dataset)
row = [1.51915,12.73,1.85,1.86,72.69,0.60,10.09,0.00,0.00]
print('>Predicted=%d (expected 3)' % (model.predict([row])))
# known class 4 (class=6 in the dataset)
row = [1.51115,17.38,0.00,0.34,75.41,0.00,6.65,0.00,0.00]
print('>Predicted=%d (expected 4)' % (model.predict([row])))
# known class 5 (class=7 in the dataset)
row = [1.51556,13.87,0.00,2.54,73.23,0.14,9.41,0.81,0.01]
print('>Predicted=%d (expected 5)' % (model.predict([row])))