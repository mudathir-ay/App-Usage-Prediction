from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import math
app = Flask(__name__)
model = pickle.load(open('./app_usage_model.pkl', 'rb'))
with open("./onehot.txt","r") as f:
    one_hot_codes= [line.strip() for line in f.readlines()]

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        day = int(request.form['day'])
        dayofweek= int(request.form['dayofweek'])
        month= int(request.form['month'])
        quarter=int(request.form['quarter'])
        dayofyear= int(request.form['dayofyear'])
        weekofyear= int(request.form['weekofyear'])
        apptype=request.form['apptype']
        input_data=[day,dayofweek,month,quarter,dayofyear,weekofyear]
        apptypelist=[0 for x in one_hot_codes]
        if apptype in one_hot_codes:
            temp=one_hot_codes.index(apptype)
            apptypelist[temp]=1
        input_data.extend(apptypelist)
        prediction=model.predict([input_data])
        output =round(prediction[0],2)
        min=int(output)
        sec=(output-min)*60
        #output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry invalid inputs")
        else:
            return render_template('index.html',prediction_text="Total minutes spent is: {} minutes and {} sec".format(min,int(sec)))
    else:
        return render_template('index.html')

if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    
