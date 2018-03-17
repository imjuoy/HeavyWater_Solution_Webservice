from flask import Flask, render_template, request, redirect, url_for, abort, session

app = Flask(__name__)
<<<<<<< HEAD
lmfit = 0
counter = 0
=======
>>>>>>> 0f01286756da0bdb182b8c2955bbb0c19924b919

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
<<<<<<< HEAD
    global counter
    stream_of_words = request.form['words']
    def initialization():
    	import numpy as np
   	import pandas as pd
    	from sklearn.feature_extraction.text import TfidfVectorizer
	data = pd.read_csv("https://s3.amazonaws.com/heavywatertest/shuffled-full-set-hashed.csv", header=None)
	data.columns = ['y', 'X']
	data = data[data.y.str.isupper()]
	vectorizer = TfidfVectorizer(min_df=1,max_features=300)
	X = vectorizer.fit_transform(data['X'].values.astype('U'))
	X_train = X.toarray()
	vals = data.y.values
	uniqueVals = np.unique(vals)
	labels = {item:index for index,item in enumerate(uniqueVals)}
	labels_opp = {index:item for index,item in enumerate(uniqueVals)}
	data['labels'] = data.y.astype('category').cat.codes
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import train_test_split
	X = X_train
	y = data['labels'].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
	c = 100
	lmfit = LogisticRegression(C=c).fit(X_train, y_train)
	yhat_lm = lmfit.predict(X_test)
	#np.mean(y_test!=yhat_lm)
    def predict(stream_of_words):
	X_new_test = words
	X_test = vectorizer.transform(X_new_test.astype('U'))
	X_test = X_test.toarray()
	#print(X_test)
	y_new_test = lmfit.predict(X_test)
	return str(y_new_test)
    if counter == 0:
	initialization()
	counter+=1
    elif counter > 0:
	result=predict(stream_of_words)
    	return render_template('result.html',result)
=======
    stream_of_words = request.form['words']
    return render_template('result.html',words = stream_of_words)
>>>>>>> 0f01286756da0bdb182b8c2955bbb0c19924b919

if __name__ == '__main__':
    app.run(host = '0.0.0.0', threaded=True, debug=True)
