import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["This is very strange",
          "This is very nice"]
vectorizer = TfidfVectorizer(min_df=1, max_features=10000)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_

data = pd.read_csv("/Volumes/Users/tusharg1/Downloads/shuffled-full-set-hashed.csv", header=None)

data.columns = ['y', 'X']

#Removing junk rows
data = data[data.y.str.isupper()]

vectorizer = TfidfVectorizer(min_df=1, max_features=10000)
X = vectorizer.fit_transform(data['X'].values.astype('U'))

X_train = X.toarray()

#Method 2: Using PCA
from sklearn.decomposition import TruncatedSVD
pca = TruncatedSVD(n_components=300)
X_reduced_train = pca.fit_transform(X_train)
X_reduced_train

vals = data.y.values
uniqueVals = np.unique(vals)
labels = {item:index for index,item in enumerate(uniqueVals)}
labels_opp = {index:item for index,item in enumerate(uniqueVals)}

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = X_train
y = data['labels'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

#c = 1000000 #No penalization
c = 100
lmfit = LogisticRegression(C=c).fit(X_train, y_train)
yhat_lm = lmfit.predict(X_test)
print(np.mean(y_test==yhat_lm))
np.mean(y_test!=yhat_lm)

from sklearn.metrics import confusion_matrix

y_test_predict = yhat_lm
cf = confusion_matrix(y_test, y_test_predict)
cfDf = pd.DataFrame(cf, columns=uniqueVals, index=uniqueVals)

