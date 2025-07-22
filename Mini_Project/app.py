from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings("ignore")

elastic = pickle.load(open("C:\\Users\\hp\\Desktop\\CODING\\Machine Learning\\ML_module_2\\Mini_Project\\Models\\model.pkl", "rb"))
scaler = pickle.load(open("C:\\Users\\hp\\Desktop\\CODING\\Machine Learning\\ML_module_2\\Mini_Project\\Models\\scaler.pkl", "rb"))

df=pd.read_csv(r"C:\Users\hp\Desktop\CODING\Machine Learning\ML_module_2\Mini_Project\EDA\dfwith encode.csv")
location_dict=df[['location','location_encode']].to_dict("split")['data'] # imppppppp
size_dict=df[['size','size_encode']].to_dict("split")['data'] # imppppppppppppppp


app=Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/main")
def main():
    return render_template("main.html")

@app.route("/predict",methods=['GET','POST'])
def predict_price():
    if request.method=='POST':
        balcony=float(request.form.get('balcony'))
        bath=float(request.form.get('bath'))
        total_sqft=float(request.form.get('total_sqft'))
        location=request.form.get('location')
        size=request.form.get('size')
        Remarks=request.form.get('Remarks')
        Price=None
        location_encode=None
        size_encode=None
        for i in location_dict:
            if i[0]==location:
                break
        location_encode=i[1]
        for i in size_dict:
            if i[0]==size:
                break
        size_encode=i[1]
        form_result=[[total_sqft,bath,balcony,location_encode,size_encode]]
        # total_sqft	bath	balcony	location_encode	size_encode
        scaled_data=scaler.transform(form_result)
        result1=elastic.predict(scaled_data)
        Price=result1
        Price=np.round(Price,2)





        # result1=(balcony,bath,total_sqft,location,size,Remarks)
        return render_template('result.html',context=[balcony,bath,total_sqft,location,size,Remarks,Price])


if __name__=='__main__':
    app.run(host='0.0.0.0')