from flask import Flask, render_template, request, jsonify
import requests
import joblib
import numpy as np
import sklearn





app = Flask(__name__)

filename = joblib.load(open("xgb.pkl", "rb"))


work=[' State-gov', ' Self-emp-not-inc' ,' Private' ,' Federal-gov', ' Local-gov','unknown' ,' Self-emp-inc' ,' Without-pay' ,' Never-worked',"<select>"]
edu = [' Bachelors', ' HS-grad' ,' 11th' ,' Masters' ,' 9th' ,' Some-college',' Assoc-acdm' ,' 7th-8th' ,' Doctorate', ' Assoc-voc' ,' Prof-school',' 5th-6th' ,' 10th', ' 1st-4th' ,' Preschool', ' 12th',"<select>"]
mar = [' Never-married' ,' Married-civ-spouse' ,' Divorced',' Married-spouse-absent', ' Separated' ,' Married-AF-spouse' ,' Widowed',"<select>"]
oc = [' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners' ,' Prof-specialty',' Other-service' ,' Sales', ' Transport-moving' ,' Farming-fishing',' Machine-op-inspct', ' Tech-support' ,' Craft-repair' ,'unknown',' Protective-serv', ' Armed-Forces', ' Priv-house-serv',"<select>"]
rel = [' Not-in-family', ' Husband' ,' Wife' ,' Own-child' ,' Unmarried',' Other-relative',"<select>"]
ra = [' White' ,' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo' ,' Other',"<select>"]
coun = [' United-States' ,' Cuba' ,' Jamaica', ' India' ,' Mexico' ,' South',' Puerto-Rico', ' Honduras', ' England' ,' Canada', ' Germany' ,' Iran',' Philippines', ' Italy' ,' Poland', ' Columbia' ,' Cambodia', ' Thailand',' Ecuador' ,' Laos', ' Taiwan', ' Haiti', ' Portugal' ,' Dominican-Republic',' El-Salvador' ,' France' ,' Guatemala' ,' China', ' Japan', ' Yugoslavia',' Peru' ,' Outlying-US(Guam-USVI-etc)' ,' Scotland' ,' Trinadad&Tobago', ' Greece' ', Nicaragua' ,' Vietnam' ,' Hong' ,' Ireland', ' Hungary',' Holand-Netherlands',"<select>"]

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html',cla = work, educ = edu, mari=mar,occ=oc,rela=rel,race=ra,coun=coun)





@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])


    workclass = request.form["workclass"]
    if(workclass == 'work'):
        workclass = 1
    else:
        workclass = 0


    education = request.form['education']
    if (education == 'edu'):
        education = 1
    else:
        education = 0


    marital_status = (request.form["marital_status"])
    
    if marital_status == "mar":
        marital_status = 1
    else:
        marital_status = 0
  

    occupation = request.form["occupation"]
    
    if occupation == "oc":
        occupation = 1
    else:
        occupation = 0
        
    relationship = request.form["relationship"]
    
    if relationship == "rel":
        relationship = 1
    else:
        relationship = 0
        
    
    race = request.form["race"]
    
    if race == "ra":
        race = 1
    else:
        race = 0
      
        
    sex = request.form["sex"]
    if sex == "Male":
        sex = 1
    else:
        sex = 0

    capital_gain = int(request.form['capital_gain'])
    
    capital_loss = int(request.form['capital_loss'])
    
    hours_per_week = int(request.form['hours_per_week'])
    
    country = request.form["country"]
    
    if country == "coun":
        country = 1
    else:
        country = 0
    
    #data = np.array([[oldpeak,ca,trtbps,slp,thall]])
    data = np.array([[age,sex,workclass , education,marital_status,occupation,relationship,race,capital_gain,capital_loss,hours_per_week,country]])
    hello = filename.predict(data)

    return render_template("result.html", prediction= hello)

if __name__ == "__main__":
    app.run(debug=True)