#step -1 # Importing flask module in the project is mandatory 
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#Step -2 Flask constructor takes the name of  
# current module (__name__) as argument.app = Flask(__name__)

app = Flask(__name__)

#Step -3 Load Trained  Model
model = pickle.load(open('pop.pkl', 'rb'))
one  = pickle.load(open('pop1.pkl', 'rb'))

# Step -4 The route() function of the Flask class is a decorator,  
# which tells the application which URL should call  
# the associated function


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #Data = (float(x) for x in request.form.values())
    age=request.form['age']
    gender=request.form['gender']
    chest=request.form['chest']
    bp=request.form['bp']
    cholestrol=request.form['cholestrol']
    FBS=request.form['FBS']
    recteg=request.form['recteg']
    personheartrate=request.form['personheartrate']
    exang=request.form['exang']
    oldpeak=request.form['oldpeak']
    slope=request.form['slope']
    ca=request.form['ca']
    thal=request.form['thal']
    
    Data={'age':[age],
          'gender':[gender],
          'chest pain':[chest],
          'Blood Pressure':[bp],
          'Cholestrol':[cholestrol],
           'FBS':[FBS],
            'recteg':[recteg],
            'personheartrate':[personheartrate],
            'exang':[exang],
             'oldpeak':[oldpeak],
             'slope':[slope],
             'ca':[ca],
             'thal':[thal]}
  
   
    
    df1 = pd.DataFrame(Data,columns=['age','gender','chest pain','Blood Pressure','Cholestrol','FBS','recteg','personheartrate','exang','oldpeak','slope','ca','thal'])
   
    x = one.transform(df1)
    print(x.shape)
    output = model.predict(x)

    if output[0]==1:
    
        myresult="Positive"
    else:
        myresult="Negative"
    
    res = myresult
    return render_template('index.html', prediction_text=' Chance of Heart Risk {}'.format(res))


# main driver function
 # run() method of Flask class runs the application  
    # on the local development server.
if __name__ == "__main__":
    app.run(debug=True)

