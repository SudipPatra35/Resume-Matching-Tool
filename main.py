from flask import Flask,render_template,redirect,request,jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)



@app.route('/')
def start():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    if request.method == "POST":
        userId=request.form["userId"]
        password=request.form["password"]
        if userId and password :
            return render_template('index.html')
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)

