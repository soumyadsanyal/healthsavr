import flask
from flask import render_template, request, redirect
from app import app
import pymysql as mdb
import health 
#import pygal
import numpy as np
import seaborn as sns


@app.route('/slides')
def index():
 return redirect("https://docs.google.com/presentation/d/1nCqFXO27j8CHvRa31K6L9KJ58wo73bqr24-And-0wDs/pub?start=true&loop=false&delayms=5000")
# return render_template("index.html", title = 'Home', user = { 'nickname': 'Kevin' },)

@app.route('/')
@app.route('/index')
@app.route('/input')
def cities_input():
  return render_template("input.html")

#@app.route('/templates/<path:path>')
#def images(path):
# fullpath="./templates/" + path
# resp=flask.make_response(open(fullpath).read())
# resp.content_type = "image/png"
# return resp
#  return render_template("input.html")


@app.route('/output')
def oop():
# if not ('meps' in locals() and 'insurance' in locals() and 'exog' in locals()):
#  (meps, exog, insurance)=health.get_persons_and_insurance()
# insurance=health.pd.read_pickle("./theinsurance.pkl")
 person={}
# person["PREGNANT_DURING_REF_PERIOD_-_RD_3/1_BUCKET"] = int(request.args.get(clean("PREGNANT_DURING_REF_PERIOD_-_RD_3/1_BUCKET")))
 person["MARITAL_STATUS-12/31/13_(EDITED/IMPUTED)_BUCKET"] = (request.args.get(clean("MARITAL_STATUS-12/31/13_(EDITED/IMPUTED)_BUCKET")))
# person["ARTHRITIS_DIAGNOSIS_(>17)_BUCKET"] = int(request.args.get(clean("ARTHRITIS_DIAGNOSIS_(>17)_BUCKET")))
 person["RACE/ETHNICITY_(EDITED/IMPUTED)_BUCKET"] = (request.args.get(clean("RACE/ETHNICITY_(EDITED/IMPUTED)_BUCKET")))
 person["CENSUS_REGION_AS_OF_12/31/13_BUCKET"] = (request.args.get(clean("CENSUS_REGION_AS_OF_12/31/13_BUCKET")))
 person["#_OUTPATIENT_DEPT_PHYSICIAN_VISITS_13_BUCKET"] = (request.args.get(clean("#_OUTPATIENT_DEPT_PHYSICIAN_VISITS_13_BUCKET")))
# person["MULT_DIAG_HIGH_BLOOD_PRESS_(>17)_BUCKET"] = int(request.args.get(clean("MULT_DIAG_HIGH_BLOOD_PRESS_(>17)_BUCKET")))
# person["#_OUTPATIENT_DEPT_PROVIDER_VISITS_13_BUCKET"] = int(request.args.get(clean("#_OUTPATIENT_DEPT_PROVIDER_VISITS_13_BUCKET")))
 person["FAMILY'S_TOTAL_INCOME_BUCKET"] = (request.args.get(clean("FAMILY'S_TOTAL_INCOME_BUCKET")))
# person["#_OFFICE-BASED_PROVIDER_VISITS_13_BUCKET"] = int(request.args.get(clean("#_OFFICE-BASED_PROVIDER_VISITS_13_BUCKET")))
# person["PERSON'S_TOTAL_INCOME_BUCKET"] = int(request.args.get(clean("PERSON'S_TOTAL_INCOME_BUCKET")))
 person["HIGH_BLOOD_PRESSURE_DIAG_(>17)_BUCKET"] = (request.args.get(clean("HIGH_BLOOD_PRESSURE_DIAG_(>17)_BUCKET")))
 person["AGE_AS_OF_12/31/13_(EDITED/IMPUTED)_BUCKET"] = (request.args.get(clean("AGE_AS_OF_12/31/13_(EDITED/IMPUTED)_BUCKET")))
 person["#_OFFICE-BASED_PHYSICIAN_VISITS_13_BUCKET"] = (request.args.get(clean("#_OFFICE-BASED_PHYSICIAN_VISITS_13_BUCKET")))
 person["ADULT_BODY_MASS_INDEX_(>17)_-_RD_5/3_BUCKET"] = (request.args.get(clean("ADULT_BODY_MASS_INDEX_(>17)_-_RD_5/3_BUCKET")))
 person["#_NIGHTS_IN_HOSP_FOR_DISCHARGES_2013_BUCKET"] = (request.args.get(clean("#_NIGHTS_IN_HOSP_FOR_DISCHARGES_2013_BUCKET")))
# person["EDUCATION_RECODE_(EDITED)_BUCKET"] = int(request.args.get(clean("EDUCATION_RECODE_(EDITED)_BUCKET")))
 person["HIGH_CHOLESTEROL_DIAGNOSIS_(>17)_BUCKET"] = (request.args.get(clean("HIGH_CHOLESTEROL_DIAGNOSIS_(>17)_BUCKET")))
 thestring=''.join('{}{}'.format(key,val) for (key,val) in person.items())
 print(thestring)
# input()
 planid1=request.args.get("plan1")
 print(planid1)
 planid2=request.args.get("plan2")
 print(planid2)
# plan1=insurance[insurance["Plan_ID_(standard_component)"]==str(planid1)].iloc[0]
# plan2=insurance[insurance["Plan_ID_(standard_component)"]==str(planid2)].iloc[0]
 (oop,billed)=health.estimate_oop_from_database(person,planid1,planid2)
# oop=compute_oop(total, num_visits=10, breakout=[[('copay',0),('coinsurance',1.0)],[('copay',0),('coinsurance',0.4)]], deductible=750, max_oop=15000)
  #call a function from a_Model package. note we are only pulling one result in the query
 return render_template("output.html",plan1_loc=(oop[planid1]), plan2_loc=(oop[planid2]), billing=billed)


@app.route('/start')
def starting():
  return render_template("start.html")



def clean(thestring):
 return ((((((((((((((thestring.replace("\'","APOS")).replace("#","NUM")).replace("+","PLUS")).replace("/","FORSLASH")).replace(">","GTHAN")).replace("<","LTHAN")).replace("(","LEFTPAR")).replace(")","RIGHTPAR")).replace("-","DASH")).replace(":","COLON")).replace(".","FULLSTOP")).replace("&","AMPERSAND")).replace("=","EQUALSIGN")).replace("%","PERCENTSIGN"))


def dirty(thestring):
 return ((((((thestring.replace("APOS","\'")).replace("NUM","#")).replace("PLUS","+")).replace("FORSLASH","/")).replace("GTHAN",">")).replace("LTHAN","<"))













