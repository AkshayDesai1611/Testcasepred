from flask import Flask, render_template, request
#import jsonify
#import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()


app = Flask(__name__)
model = pickle.load(open('testcasepredrf.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('testpred.html')

standard_to = StandardScaler()

@app.route("/predict", methods = ['POST'])
def predict():
    if request.method == 'POST':
        Project_Size = request.form['Project_Size']
        if (Project_Size=='Small'):
            Project_Size=2
        elif (Project_Size=='Medium'):
            Project_Size=1
        else:
            Project_Size=0
        Project_Complexity = request.form['Project_Complexity']
        if (Project_Complexity=='Simple'):
            Project_Complexity=2
        elif (Project_Complexity=='Medium'):
            Project_Complexity=1
        else:
            Project_Complexity=0
        Requirements = int(request.form['Requirements'])

        Systems_with_func_change = int(request.form['Systems_with_func_change'])
        Systems_with_config_change = int(request.form['Systems_with_config_change'])
        Systems_with_no_change = int(request.form['Systems_with_no_change'])
        
        data = [Project_Size,Project_Complexity,Requirements,Systems_with_func_change,Systems_with_config_change,Systems_with_no_change]
        data = lb.fit_transform(data)
        prediction=model.predict([data])
        output=round(prediction[0],2)
        if output<0:
            return render_template('testpred.html',prediction_texts="Sorry please check your inputs")
        else:
            return render_template('testpred.html',prediction_text="The test-cases are {}".format(output))
    else:
        return render_template('testpred.html')

if __name__=="__main__":
    app.run(debug=True)

