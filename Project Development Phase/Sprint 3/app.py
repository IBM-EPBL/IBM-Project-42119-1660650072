#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle as pk


# In[2]:


app=Flask(__name__)
model=pk.load(open('CKD.pkl','rb'))

@app.route('/')
def home():
    return render_template('homepage.html')
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
	return render_template('indexpage.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
	return render_template('homepage.html')
@app.route('/predict',methods=['POST'])
def predict():
    input_features=[float(x) for x in request.form.values()]
    features_value=[np.array(input_features)]
    features_name=['blood_urea','blood_glucose_random','coronary_artery_disease','anemia','pus_cell','red_blood_cells','diabetesmellitus','pedal_edema']
    df=pd.DataFrame(features_value,columns=features_name)
    output=model.predict(df)
    if(output==1):
        return render_template('predictionNo.html')
    else:
        return render_template('predictionYes.html')


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




