import pandas as pd

vowel_train = pd.read_csv('vowel.train.csv', sep=',', header=0)
vowel_test = pd.read_csv('vowel.test.csv', sep=',', header=0)

vowel_train.head()

y_tr = vowel_train.iloc[:,0]
X_tr = vowel_train.iloc[:,1:]

y_test = vowel_test.iloc[:,0]
X_test = vowel_test.iloc[:,1:]

print(X_test)
print(y_test)