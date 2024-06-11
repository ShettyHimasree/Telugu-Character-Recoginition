import numpy as np
from flask import Flask,request,jsonify,render_template,redirect
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import pickle
import pandas as pd
import statistics as st
from scipy.stats import kurtosis,entropy
import chardet
import itertools
app=Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = '.'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Predict")
 # load the pickle model

model=pickle.load(open("RandomForest.pkl","rb"))


@app.route("/",methods=['GET',"POST"])
def Home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        if os.path.exists(os.path.join(os.getcwd(),"test.csv")):
            os.remove("test.csv")

        os.rename(file.filename,"test.csv")

        ##preprocessing
        def sign_change(x):
            return len(list(itertools.groupby(x, lambda x: x > 0)))
        file_path=os.path.join(os.getcwd(),"test.csv")
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        print(result['encoding'],file_path)
        df=pd.read_csv(file_path,delimiter='\t',encoding=result['encoding'],skiprows=3)
        df.to_csv(os.path.join(os.getcwd(),"test1.csv"),index=False)

        file_path=os.path.join(os.getcwd(),"test1.csv")
        df=pd.read_csv(file_path)
        li=[]
        df=df.iloc[:,1:4]
        print(df)
        skew=list(df.skew(axis=0))
        k=0

        for i in df.columns:
            li.append(np.mean(df[i]))
            li.append(np.median(df[i]))
            li.append(st.mode(df[i]))
            li.append(max(df[i]))
            li.append(min(df[i]))
            li.append(np.var(df[i]))
            li.append(np.std(df[i]))
            li.append(np.sqrt(np.mean(df[i]*df[i])))
            li.append(skew[k])
            k+=1
            li.append(kurtosis(df[i]))
            #             li.append(entropy(a5[i]))
            li.append(sum(df[i]))
            li.append(sign_change(df[i]))

        cn=[]
        for i in range(len(li)):
            cn.append(i)

        df = pd.DataFrame(columns=cn)
        df = df._append(pd.Series(li, index=df.columns), ignore_index=True)
        df.to_csv(os.path.join(os.getcwd(),"test1.csv"),index=False)

##
        dft = pd.read_csv(r'test1.csv')
        print(dft)
        x=dft.iloc[:,:].values
        prediction=model.predict(x)
        dic={
            1:"అ",
            2:"ఆ",
            3:"ఇ",
            4:"ఈ",
            5:"ఉ"
        }
        dic1={
            1:"https://res.cloudinary.com/deo6qmwsc/image/upload/v1695700350/telugu%20characters/g2ljbtadebhrai85fexb.png",
            2:"https://res.cloudinary.com/deo6qmwsc/image/upload/v1695700350/telugu%20characters/kvn2r4icfeyjzqygbiny.png",
            3:"https://res.cloudinary.com/deo6qmwsc/image/upload/v1695700350/telugu%20characters/xljw52bv4m0cppghpi6k.png",
            4:"https://res.cloudinary.com/deo6qmwsc/image/upload/v1695700350/telugu%20characters/dul1enu400yvwe3tty4g.png",
            5:"https://res.cloudinary.com/deo6qmwsc/image/upload/v1695700350/telugu%20characters/jegqqbbx5gk1g8s7juqs.png"
        }
        print(prediction[0])
        os.remove("test.csv")
        return render_template("index.html", prediction_text = "{}".format(dic1[int(prediction[0])]),form=form)

        # return redirect("/predict")
    return render_template('index.html',prediction_text ="https://i7x7p5b7.stackpathcdn.com/codrops/wp-content/uploads/2015/09/smart-custom-file-input-2.gif?x39121",form=form)



if __name__=="__main__":
    app.run(debug=True)
