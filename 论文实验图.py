import numpy as np
import matplotlib as plt
import pandas as pd
import os
path='data'+os.sep+'demo.txt'
pdData=pd.read_csv(path,header=None,names=['learning_rate','recall','precious'])
recall=pdData[pdData['recall']==1]
precious=pdData[pdData['preci']==0]
fig,ax=plt.subplots(figsize=(10,5))
ax.scatter(positive['Exam 1'],positive['Exam 2'],s=30,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam 1'],negative['Exam 2'],s=30,c='r',marker='x',label='No Admitted')