from flask import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

@app.route('/')       #### default route
def index():
    return render_template('index.html')

##########################################################
url="https://raw.githubusercontent.com/sarwansingh/Python/master/ClassExamples/data/diamond.csv"
df=pd.read_csv(url)
df.drop('Unnamed: 0', axis=1, inplace=True)
df.cut.replace(['Premium', 'Ideal','Good','Very Good','Fair'], [1, 2,3,4,5], inplace=True)
df.color.replace(['D','E','F','G','H','I','J'], [1,2,3,4,5,6,7], inplace=True)
df.clarity.replace(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], [1,2,3,4,5,6,7,8], inplace=True)

df = df.drop(df[df["x"]==0].index)
df = df.drop(df[df["y"]==0].index)
df= df.drop(df[df["z"]==0].index)

#Dropping the outliers.
df = df[(df["depth"]<75)&(df["depth"]>45)]
df = df[(df["table"]<80)&(df["table"]>40)]
df = df[(df["x"]<30)]
df = df[(df["y"]<30)]
df = df[(df["z"]<30)&(df["z"]>2)]

X= df.drop(["price"],axis =1)
Y= df["price"]

#pipeline_rf=Pipeline([("scalar3",StandardScaler()),"rf_classifier",RandomForestRegressor())])
#pipeline_rf.fit(X,Y)
model_1=RandomForestRegressor()
model_1.fit(X,Y)

#0.31	3	1	2	63.3	58.0	335	4.34	4.35	2.75
pred=model_1.predict([[0.31,3,1,2,63.3,58.0,4.34,	4.35,2.75]])
op_1=str(round(pred[0])) + ' la'
op_1

##########################################################

@app.route("/project")
def dproject():
  return render_template("form.html")

@app.route("/predict",methods=["POST"])
def dpredict():
  carat=float(request.form["carat"])
  cut=int(request.form["cut"])
  color=int(request.form["color"])
  clarity=int(request.form["clarity"])
  depth=float(request.form["depth"])
  table=float(request.form["table"])
  x=float(request.form["x"])
  y=float(request.form["y"])
  z=float(request.form["z"])
  res_1=model_1.predict([[carat,cut,color,clarity,depth,table,x,y,z]])
  op_1= "      Predicted Price: " +str(round(res_1[0]))+" lakh"
  return render_template("form.html",result_1=op_1)

if __name__=="__main__":
  app.run()
  

  
  
  
  