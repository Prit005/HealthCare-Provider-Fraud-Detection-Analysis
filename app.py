from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
print(model)

@app.route('/',methods=['GET'])
def Home():
    return render_template('deploy.html')

@app.route("/predict", methods=['POST'])
def predict():
    c=['Provider', 'BeneID', 'Gender', 'Race',
        'RenalDiseaseIndicator', 'State', 'County', 'ChronicCond_Alzheimer',
        'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
        'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
        'ChronicCond_Depression', 'ChronicCond_Diabetes',
        'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
        'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke',
        'InscClaimAmtReimbursed','AttendingPhysician', 
        'OperatingPhysician', 'OtherPhysician',
        'ClmAdmitDiagnosisCode', 'DeductibleAmtPaid',
        'DiagnosisGroupCode','Age_Of_Patients', 'Hospital_stay_Duration',
        'Claim_Duration','Amount_left_reimburse',
        'Diagnosis_Code_4019', 'Diagnosis_Code_25000',	
        'Diagnosis_Code_2724', 'Diagnosis_Code_V5869',
       	'Diagnosis_Code_4011', 'Diagnosis_Code_42731',	
        'Diagnosis_Code_V5861', 'Diagnosis_Code_2720',	
        'Diagnosis_Code_2449', 'Diagnosis_Code_4280',	
        'Procedure_Code_4019.0', 'Procedure_Code_9904.0',	
        'Procedure_Code_2724.0', 'Procedure_Code_8154.0',
        'Procedure_Code_66.0', 'Procedure_Code_3893.0',
        'Procedure_Code_3995.0','Procedure_Code_4516.0',	
        'Procedure_Code_3722.0', 'Procedure_Code_8151.0']  
    
    if request.method == 'POST': 
        Provider=request.form['Provider']
        BeneID=request.form['BeneID']
        Gender=int(request.form['Gender'])
        Race=int(request.form['Race'])
        RenalDiseaseIndicator=request.form['RenalDiseaseIndicator']
        State=int(request.form['State'])
        County=int(request.form['County'])
        ChronicCond_Alzheimer=request.form['ChronicCond_Alzheimer']
        ChronicCond_Heartfailure=request.form['ChronicCond_Heartfailure']
        ChronicCond_KidneyDisease=request.form['ChronicCond_KidneyDisease']
        ChronicCond_Cancer=request.form['ChronicCond_Cancer']
        ChronicCond_ObstrPulmonary=request.form['ChronicCond_ObstrPulmonary']
        ChronicCond_Depression=request.form['ChronicCond_Depression']
        ChronicCond_Diabetes=request.form['ChronicCond_Diabetes']
        ChronicCond_IschemicHeart=request.form['ChronicCond_IschemicHeart']
        ChronicCond_Osteoporasis=request.form['ChronicCond_Osteoporasis']
        ChronicCond_rheumatoidarthritis=request.form['ChronicCond_rheumatoidarthritis']
        ChronicCond_stroke=request.form['ChronicCond_stroke']
        InscClaimAmtReimbursed=int(request.form['InscClaimAmtReimbursed'])
        AttendingPhysician=request.form['AttendingPhysician']
        OperatingPhysician=request.form['OperatingPhysician']
        OtherPhysician=request.form['OtherPhysician']
        DeductibleAmtPaid=request.form['DeductibleAmtPaid']
        ClmAdmitDiagnosisCode=request.form['ClmAdmitDiagnosisCode']
        DiagnosisGroupCode=request.form['DiagnosisGroupCode']
        Age_Of_Patients=request.form['Age_Of_Patients']
        Hospital_stay_Duration=request.form["Hospital_stay_Duration"]
        Claim_Duration=request.form['Claim_Duration']
        Amount_left_reimburse=request.form['Amount_left_reimburse']
        Diagnosis_Code_1=request.form['Diagnosis_Code_4019']
        Diagnosis_Code_2=request.form['Diagnosis_Code_25000']
        Diagnosis_Code_3=request.form['Diagnosis_Code_2724']
        Diagnosis_Code_4=request.form['Diagnosis_Code_V5869']
        Diagnosis_Code_5=request.form['Diagnosis_Code_4011']
        Diagnosis_Code_6=request.form['Diagnosis_Code_42731']
        Diagnosis_Code_7=request.form['Diagnosis_Code_V5861']
        Diagnosis_Code_8=request.form['Diagnosis_Code_2720']
        Diagnosis_Code_9=request.form['Diagnosis_Code_2449']
        Diagnosis_Code_10=request.form['Diagnosis_Code_4280']
        Procedure_Code_1=request.form['Procedure_Code_4019.0']
        Procedure_Code_2=request.form['Procedure_Code_9904.0']
        Procedure_Code_3=request.form['Procedure_Code_2724.0']
        Procedure_Code_4=request.form['Procedure_Code_8154.0']
        Procedure_Code_5=request.form['Procedure_Code_66.0']
        Procedure_Code_6=request.form['Procedure_Code_3893.0']
        Procedure_Code_7=request.form['Procedure_Code_3995.0']
        Procedure_Code_8=request.form['Procedure_Code_4516.0']
        Procedure_Code_9=request.form['Procedure_Code_3722.0']
        Procedure_Code_10=request.form['Procedure_Code_8151.0']
        
        import pandas as pd

        data = [Provider, BeneID, Gender, Race,
        RenalDiseaseIndicator, State, County, ChronicCond_Alzheimer,
        ChronicCond_Heartfailure, ChronicCond_KidneyDisease,
        ChronicCond_Cancer, ChronicCond_ObstrPulmonary,
        ChronicCond_Depression, ChronicCond_Diabetes,
        ChronicCond_IschemicHeart, ChronicCond_Osteoporasis,
        ChronicCond_rheumatoidarthritis, ChronicCond_stroke,
        InscClaimAmtReimbursed,AttendingPhysician, OperatingPhysician, OtherPhysician,
        ClmAdmitDiagnosisCode, DeductibleAmtPaid,
        DiagnosisGroupCode,Age_Of_Patients,Hospital_stay_Duration,
        Claim_Duration, Amount_left_reimburse, 
        Diagnosis_Code_1, Diagnosis_Code_2,	
        Diagnosis_Code_3, Diagnosis_Code_4,
       	Diagnosis_Code_5, Diagnosis_Code_6,
        Diagnosis_Code_7, Diagnosis_Code_8, Diagnosis_Code_9,
        Diagnosis_Code_10, Procedure_Code_1,
        Procedure_Code_2, Procedure_Code_3,
        Procedure_Code_4, Procedure_Code_5, 
        Procedure_Code_6, Procedure_Code_7, 
        Procedure_Code_8, Procedure_Code_9, Procedure_Code_10]
        
        import numpy as np
        data=np.asarray(data)
        df = pd.DataFrame([data], columns=c)
        
        Provider=load(open("dictionary_object/Provider_dict.json","rb"))
        BeneID=load(open("dictionary_object/BeneID_dict.json","rb"))
        AttendingPhysician=load(open("dictionary_object/AttendingPhysician_dict.json","rb"))
        OperatingPhysician=load(open("dictionary_object/OperatingPhysician_dict.json","rb"))
        OtherPhysician=load(open("dictionary_object/OtherPhysician_dict.json","rb"))
        ClmAdmitDiagnosisCode=load(open("dictionary_object/ClmAdmitDiagnosisCode_dict.json","rb"))
        DiagnosisGroupCode=load(open("dictionary_object/DiagnosisGroupCode_dict.json","rb")) 
        
        # replacing categories to count for whole dataset
        def replace_id(data,column):
            ''' Categories to count and then stored to dictionary as Key: Value
                    After that we map that in particular column and replace  
            '''
            value_count = data[column].value_counts().to_dict()
    
            data[column] = data[column].map(value_count)
            return data[column]

        # Provider
        replace_id(df,"Provider")

        # BeneID
        replace_id(df,"BeneID")

        # AttendingPhysician
        replace_id(df,"AttendingPhysician")
   
        # OperatingPhysician
        replace_id(df,"OperatingPhysician")

        # OtherPhysician
        replace_id(df,"OtherPhysician")

        # ClmAdmitDiagnosisCode
        replace_id(df,"ClmAdmitDiagnosisCode")
  
        # DiagnosisGroupCode
        replace_id(df,"DiagnosisGroupCode")
        
         # Normalizing data
        scale_columns = ["Provider","BeneID","Race","State","County","InscClaimAmtReimbursed",
                   "AttendingPhysician","OperatingPhysician","OtherPhysician",
                   "ClmAdmitDiagnosisCode","DeductibleAmtPaid","DiagnosisGroupCode",
                   "Age_Of_Patients","Hospital_stay_Duration","Claim_Duration","Amount_left_reimburse"]

        for i in scale_columns:    
            scale = load('scale_cols/'+i+'_std_scaler.bin')
            scale.clip = False
            df[i] = scale.transform(df[i].values.reshape(-1,1))
        
        prediction = model.predict(df)

        if prediction[0]==1:
            return render_template('deploy.html',prediction_texts="FRAUDULENT CLAIM")
        else:
            return render_template('deploy.html',prediction_text="NON FRAUDULENT CLAIM")
     
if __name__=="__main__":
    app.run(debug=True)

