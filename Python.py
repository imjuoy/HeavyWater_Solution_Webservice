from flask import Flask, render_template, request, redirect, url_for, abort, session

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    stream_of_words = request.form['words']
    return render_template('result.html',words = stream_of_words)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', threaded=True, debug=True)
