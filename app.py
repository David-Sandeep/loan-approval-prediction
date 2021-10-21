from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("xgb_model.pkl", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():

    """   ['Credit_History', 'LoanAmount','Total_Income','Loan_Amount_Term' 
        'Gender', 'Married',
       'Dependents',
       'Education, 'Self_Employed',
       'Property_Area',
       ]

        ['Credit_History', 'LoanAmount_log', 'Gender_Male', 'Married_Yes',
        'Dependents_0', 'Dependents_1', 'Dependents_2',
        'Education_Not Graduate', 'Self_Employed_Yes',
        'Property_Area_Semiurban', 'Property_Area_Urban', 'Total_Income_log',
        'EMI', 'Balance Income'] """

    if request.method == "POST":

        # Credit_History
        Credit_History = int(request.form["Credit_History"])

        # LoanAmount
        LoanAmount = int(request.form["LoanAmount"])

        # Total_Income
        Total_Income = int(request.form["Total_Income"])

        # Loan_Amount_Term
        Loan_Amount_Term = int(request.form["Loan_Amount_Term"])

        #Gender
        Gender=request.form['Gender']
        if(Gender=='male'):
            Gender_Male=1
        else:
            Gender_Male=0

        #Married
        Married=request.form['Married']
        if(Married=='yes'):
            Married_Yes=1
        else:
            Married_Yes=0

        #Dependents
        Dependents=request.form['Dependents']
        if Dependents=='0':
            Dependents_0=1
            Dependents_1=0
            Dependents_2=0
        elif Dependents=='1':
            Dependents_0=0
            Dependents_1=1
            Dependents_2=0

        elif Dependents=='2':
            Dependents_0=0
            Dependents_1=0
            Dependents_2=1
        else:
            Dependents_0=0
            Dependents_1=0
            Dependents_2=0

        #Education
        Education=request.form['Education']
        if Education=='Not Graduate':
            Education_Not_Graduate=1
           
        else:
            Education_Not_Graduate=0

        #Self_Employed
        Self_Employed=request.form['Self_Employed']
        if Self_Employed=='yes':
            Self_Employed_Yes=1
           
        else:
            Self_Employed_Yes=0

        #Property_Area
        Property_Area=request.form['Property_Area']
        if Property_Area=='Semiurban':
            Property_Area_Semiurban=1
            Property_Area_Urban=0
        elif Property_Area=='Urban':
            Property_Area_Semiurban=0
            Property_Area_Urban=1
        else:
            Property_Area_Semiurban=0
            Property_Area_Urban=0

        LoanAmount_log=np.log(LoanAmount)
        Total_Income_log=np.log(Total_Income)
        EMI=LoanAmount/Loan_Amount_Term
        Balance_Income=Total_Income - (EMI*1000)


        fs=[Credit_History, LoanAmount_log, Gender_Male, Married_Yes,
       Dependents_0, Dependents_1, Dependents_2,
       Education_Not_Graduate, Self_Employed_Yes,
       Property_Area_Semiurban, Property_Area_Urban, Total_Income_log,
       EMI, Balance_Income]

        print(fs)
        prediction=model.predict(np.array(fs).reshape(1,14))

        print(prediction)

        
        if prediction[0]==1:
            result='Result:you can get loan!'
        else:
            result='Result:sorry you cannot get loan!'

        return render_template('home.html',prediction_text=" {}".format(result))


    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)
