import pickle

from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def hello_world():
    return render_template("PredictionPage.html")


@app.route('/xyz', methods=['POST', 'GET'])
def xyz():
  if request.method == 'POST':
     data1 = request.form['Present Price']
     data2 = request.form['Nth Owner']
     data3 = request.form['Car Age']
     #data4 = request.form.get('Diesel/Petrol(1/0)',False)
     #data5 = request.form.get('Manual/Automatic(1/0)',False)
     data4 = request.form['Diesel/Petrol(0/1)']
     data5 = request.form['Manual/Automatic(1/0)']


     array1 = np.array([[data1, data2, data3, data4,data5]],dtype=np.float_)

     value = model.predict(array1)
     return render_template("PredictionPage.html", pred="The selling value is Rs {} lakhs".format(value))
  else:
     return render_template("After.html")

if __name__ == '__main__':
    app.run(debug = True)
