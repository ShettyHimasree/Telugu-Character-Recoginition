import pandas as pd
import numpy as np
import statistics as st
from scipy.stats import kurtosis,entropy
import os
import chardet
import itertools

def sign_change(x):
    return len(list(itertools.groupby(x, lambda x: x > 0)))
file_path=os.path.join(os.getcwd(),"test.csv")
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())
print(result['encoding'],file_path)
df=pd.read_csv(file_path,delimiter='\t',encoding=result['encoding'],skiprows=3)
print(df)
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
print(cn)
print(df)

