import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs

import plotly.io as pio
import plotly.express as px
from plotly.offline import plot
import seaborn as sns
import datetime as dt
import warnings
plt.style.use('dark_background')
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv('Life_Expectancy_Data.csv')

df.head()

df.isnull().sum()

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',fill_value=None)
df['Life expectancy ']=imputer.fit_transform(df[['Life expectancy ']])
df['Adult Mortality']=imputer.fit_transform(df[['Adult Mortality']])
df['Alcohol']=imputer.fit_transform(df[['Alcohol']])
df['Hepatitis B']=imputer.fit_transform(df[['Hepatitis B']])
df[' BMI ']=imputer.fit_transform(df[[' BMI ']])
df['Polio']=imputer.fit_transform(df[['Polio']])
df['Total expenditure']=imputer.fit_transform(df[['Total expenditure']])
df['Diphtheria ']=imputer.fit_transform(df[['Diphtheria ']])
df['GDP']=imputer.fit_transform(df[['GDP']])
df['Population']=imputer.fit_transform(df[['Population']])
df[' thinness  1-19 years']=imputer.fit_transform(df[[' thinness  1-19 years']])
df[' thinness 5-9 years']=imputer.fit_transform(df[[' thinness 5-9 years']])
df['Income composition of resources']=imputer.fit_transform(df[['Income composition of resources']])
df['Schooling']=imputer.fit_transform(df[['Schooling']])


df.isnull().sum()

df.describe()

df.head()

df.corr().head()


#plots: ( all plots commands are commeneted but you can choose to uncomment the ones you wish to create 

#plt.figure(figsize=(15,10))
#sns.heatmap(df.corr(),annot=True,cmap='Reds')
#plt.show()

# hist_plot=px.histogram(df,x='Life expectancy ',template='plotly_dark')
# plot(hist_plot)

# fig=px.violin(df,x='Status',y='Life expectancy ',color='Status',template='plotly_dark',box=True,title='Life expectancy Based on Countries status')
# plot(fig)


# fig=px.line(df.sort_values(by='Year'),x='Year',y='Life expectancy ',animation_frame='Country',animation_group='Year',color='Country',markers=True,template='plotly_dark',title='<b> Country wise Life Expectancy over Years')
# plot(fig)

# plot(px.scatter(df,x='Life expectancy ',y='Life expectancy ',color='Country',size='Year',template='plotly_dark',title='<b> Life Expectancy Versus Percentage expenditure'))