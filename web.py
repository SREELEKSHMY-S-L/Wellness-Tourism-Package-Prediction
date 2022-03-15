from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
   age = float(request.values['Age'])
  
   monthly_income = float(request.values['Monthly Income'])
   duration = float(request.values['duration'])
   passport = float(request.values['Passport']) 
   number_of_trips = float(request.values['No. of Trips'])
   input_data = np.array([[age, monthly_income,duration, number_of_trips, passport]])
   
   output=model.predict(input_data)
   output=output.item()
   if output == 1:
       return render_template ('result.html', prediction_text="The client is likely to purchase the package")
   else:
       return render_template ('result.html', prediction_text="The client is not likely to purchase the package")
if __name__=='__main__':
    app.run(port=8000)

