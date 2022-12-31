import HeartLR
import HeartKNN
import HeartLOGR
import HeartRFC
from flask import Flask, render_template
posts=[
    {
        'model': 'Linear Regression',
        'Accuracy': HeartLR.rmse,
        'Type': 'It is Good for Linear data an not for classification data, as it predicts suitable linear function for the given problem'
    },
    {
        'model': 'Logistic Regression',
        'Accuracy': HeartLOGR.LogRacc,
        'Type': 'It is Good for Classification data and not for Linear data, as it predicts suitable non linear function for the given problem'
    },
    {
        'model': 'K Nearest Neighbour Regression',
        'Accuracy': HeartKNN.HKNNaccuracy,
        'Type': 'It is Good for Classification data, because it chooses n number of points and the closest distance between then is the class it belongs to.'
    },
    {
        'model': 'Random Forest Generator',
        'Accuracy': HeartRFC.HRFCacc,
        'Type': 'It is Good for Classification data,Because it divides the data perfectly on the basis of their attributs by applying different conditions, until the best fit is obtained.'
    }
       
]
app = Flask(__name__)
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return "<h1><center>About The Page</center></h1>"

app.run(debug=True) 