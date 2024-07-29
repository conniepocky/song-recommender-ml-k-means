from flask import Flask, render_template
from main import recommend

app = Flask(__name__)

@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'

@app.route('/<name>')
def hello_name(name):
   songs = recommend(name)
   print(songs)
   return render_template("index.html", songs=songs)

if __name__ == '__main__':
   app.run()