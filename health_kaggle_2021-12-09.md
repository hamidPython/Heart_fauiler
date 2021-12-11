```python
'''
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million 
lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be 
used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use,
unhealthy diet and obesity,
physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk 
(due to the presence of one or more risk factors such as hypertension, diabetes, 
hyperlipidaemia or already established disease) need early detection and management wherein 
a machine learning model can be of great help.


'''
```




    '\nCardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million \nlives each year, which accounts for 31% of all deaths worlwide.\nHeart failure is a common event caused by CVDs and this dataset contains 12 features that can be \nused to predict mortality by heart failure.\n\nMost cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use,\nunhealthy diet and obesity,\nphysical inactivity and harmful use of alcohol using population-wide strategies.\n\nPeople with cardiovascular disease or who are at high cardiovascular risk \n(due to the presence of one or more risk factors such as hypertension, diabetes, \nhyperlipidaemia or already established disease) need early detection and management wherein \na machine learning model can be of great help.\n\n\n'



Attribute Information:

Thirteen (13) clinical features:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)-کم خونی
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)---10 to 120 micrograms per liter (mcg/L)

- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)---EF  در قلب نرمال بین 55 تا 70 درصد می تواند باشد
- platelets: platelets in the blood (kiloplatelets/mL)--۱۵۰ تا ۴۵۰ هزار عدد در هر
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)--جواب آزمایش creatinine ( کراتین ): مقدار طبیعی کراتینین خون در مردان ۰٫۸-۱٫۲ mg/dl (در برخی موارد تا ۱٫۵ mg/dl ) و در زنان ۰٫۶-۰٫۹ mg/dl می باشد.

- serum sodium: level of serum sodium in the blood (mEq/L)


مقادیر نرمال
نوزادان : 134-144 mEq/L
اطفال : 134-150 mEq/L
کودکان : 136-145 mEq/L
بزرگسالان : 136-145 mEq/L


- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [target] death event: if the patient deceased during the follow-up period (boolean)


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import re
```


```python
df = pd.read_csv('heart.csv')
```


```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>anaemia</th>
      <th>creatinine_phosphokinase</th>
      <th>diabetes</th>
      <th>ejection_fraction</th>
      <th>high_blood_pressure</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
      <th>sex</th>
      <th>smoking</th>
      <th>time</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>75.0</td>
      <td>0</td>
      <td>582</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>265000.00</td>
      <td>1.9</td>
      <td>130</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>55.0</td>
      <td>0</td>
      <td>7861</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>263358.03</td>
      <td>1.1</td>
      <td>136</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>65.0</td>
      <td>0</td>
      <td>146</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>162000.00</td>
      <td>1.3</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>50.0</td>
      <td>1</td>
      <td>111</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>210000.00</td>
      <td>1.9</td>
      <td>137</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>65.0</td>
      <td>1</td>
      <td>160</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>327000.00</td>
      <td>2.7</td>
      <td>116</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
           'ejection_fraction', 'high_blood_pressure', 'platelets',
           'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
           'DEATH_EVENT'],
          dtype='object')




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 299 entries, 0 to 298
    Data columns (total 13 columns):
    age                         299 non-null float64
    anaemia                     299 non-null int64
    creatinine_phosphokinase    299 non-null int64
    diabetes                    299 non-null int64
    ejection_fraction           299 non-null int64
    high_blood_pressure         299 non-null int64
    platelets                   299 non-null float64
    serum_creatinine            299 non-null float64
    serum_sodium                299 non-null int64
    sex                         299 non-null int64
    smoking                     299 non-null int64
    time                        299 non-null int64
    DEATH_EVENT                 299 non-null int64
    dtypes: float64(3), int64(10)
    memory usage: 30.5 KB
    


```python
##There is no null.
df.isnull().sum()
```




    age                         0
    anaemia                     0
    creatinine_phosphokinase    0
    diabetes                    0
    ejection_fraction           0
    high_blood_pressure         0
    platelets                   0
    serum_creatinine            0
    serum_sodium                0
    sex                         0
    smoking                     0
    time                        0
    DEATH_EVENT                 0
    dtype: int64




```python
int_= df.select_dtypes(include='int64')
fl_= df.select_dtypes(include='float')
```


```python
int_.columns
```




    Index(['anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
           'high_blood_pressure', 'serum_sodium', 'sex', 'smoking', 'time',
           'DEATH_EVENT'],
          dtype='object')




```python
fl_.columns
```




    Index(['age', 'platelets', 'serum_creatinine'], dtype='object')




```python
con = ['age', 'platelets', 'serum_creatinine','serum_sodium']
```


```python
df[con].hist(bins=20, figsize=(10, 10))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000002D54E12AEC8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002D54E183548>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000002D54E1BF548>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002D54E1FCE88>]],
          dtype=object)




![png](output_12_1.png)



```python
df[con].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.00000</td>
      <td>299.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>60.833893</td>
      <td>263358.029264</td>
      <td>1.39388</td>
      <td>136.625418</td>
    </tr>
    <tr>
      <td>std</td>
      <td>11.894809</td>
      <td>97804.236869</td>
      <td>1.03451</td>
      <td>4.412477</td>
    </tr>
    <tr>
      <td>min</td>
      <td>40.000000</td>
      <td>25100.000000</td>
      <td>0.50000</td>
      <td>113.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>51.000000</td>
      <td>212500.000000</td>
      <td>0.90000</td>
      <td>134.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>60.000000</td>
      <td>262000.000000</td>
      <td>1.10000</td>
      <td>137.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>70.000000</td>
      <td>303500.000000</td>
      <td>1.40000</td>
      <td>140.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>95.000000</td>
      <td>850000.000000</td>
      <td>9.40000</td>
      <td>148.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['sex1']=df.sex.replace({1:'Male',0:'Female'})
```


```python
df['Marg'] = df.DEATH_EVENT.replace({1:'yes',0:'No'})
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>anaemia</th>
      <th>creatinine_phosphokinase</th>
      <th>diabetes</th>
      <th>ejection_fraction</th>
      <th>high_blood_pressure</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
      <th>sex</th>
      <th>smoking</th>
      <th>time</th>
      <th>DEATH_EVENT</th>
      <th>sex1</th>
      <th>Marg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>75.0</td>
      <td>0</td>
      <td>582</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>265000.00</td>
      <td>1.9</td>
      <td>130</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>Male</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>1</td>
      <td>55.0</td>
      <td>0</td>
      <td>7861</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>263358.03</td>
      <td>1.1</td>
      <td>136</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>Male</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>2</td>
      <td>65.0</td>
      <td>0</td>
      <td>146</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>162000.00</td>
      <td>1.3</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>Male</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>3</td>
      <td>50.0</td>
      <td>1</td>
      <td>111</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>210000.00</td>
      <td>1.9</td>
      <td>137</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>Male</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>4</td>
      <td>65.0</td>
      <td>1</td>
      <td>160</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>327000.00</td>
      <td>2.7</td>
      <td>116</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>Female</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>294</td>
      <td>62.0</td>
      <td>0</td>
      <td>61</td>
      <td>1</td>
      <td>38</td>
      <td>1</td>
      <td>155000.00</td>
      <td>1.1</td>
      <td>143</td>
      <td>1</td>
      <td>1</td>
      <td>270</td>
      <td>0</td>
      <td>Male</td>
      <td>No</td>
    </tr>
    <tr>
      <td>295</td>
      <td>55.0</td>
      <td>0</td>
      <td>1820</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>270000.00</td>
      <td>1.2</td>
      <td>139</td>
      <td>0</td>
      <td>0</td>
      <td>271</td>
      <td>0</td>
      <td>Female</td>
      <td>No</td>
    </tr>
    <tr>
      <td>296</td>
      <td>45.0</td>
      <td>0</td>
      <td>2060</td>
      <td>1</td>
      <td>60</td>
      <td>0</td>
      <td>742000.00</td>
      <td>0.8</td>
      <td>138</td>
      <td>0</td>
      <td>0</td>
      <td>278</td>
      <td>0</td>
      <td>Female</td>
      <td>No</td>
    </tr>
    <tr>
      <td>297</td>
      <td>45.0</td>
      <td>0</td>
      <td>2413</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>140000.00</td>
      <td>1.4</td>
      <td>140</td>
      <td>1</td>
      <td>1</td>
      <td>280</td>
      <td>0</td>
      <td>Male</td>
      <td>No</td>
    </tr>
    <tr>
      <td>298</td>
      <td>50.0</td>
      <td>0</td>
      <td>196</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>395000.00</td>
      <td>1.6</td>
      <td>136</td>
      <td>1</td>
      <td>1</td>
      <td>285</td>
      <td>0</td>
      <td>Male</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>299 rows × 15 columns</p>
</div>




```python
sns.pairplot(df[['creatinine_phosphokinase','ejection_fraction','platelets','serum_sodium','Marg']],hue='Marg')


```




    <seaborn.axisgrid.PairGrid at 0x2d54e5a7248>




![png](output_17_1.png)



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 299 entries, 0 to 298
    Data columns (total 15 columns):
    age                         299 non-null float64
    anaemia                     299 non-null int64
    creatinine_phosphokinase    299 non-null int64
    diabetes                    299 non-null int64
    ejection_fraction           299 non-null int64
    high_blood_pressure         299 non-null int64
    platelets                   299 non-null float64
    serum_creatinine            299 non-null float64
    serum_sodium                299 non-null int64
    sex                         299 non-null int64
    smoking                     299 non-null int64
    time                        299 non-null int64
    DEATH_EVENT                 299 non-null int64
    sex1                        299 non-null object
    Marg                        299 non-null object
    dtypes: float64(3), int64(10), object(2)
    memory usage: 35.2+ KB
    


```python
df[['sex1','anaemia','diabetes','smoking','Marg']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex1</th>
      <th>anaemia</th>
      <th>diabetes</th>
      <th>smoking</th>
      <th>Marg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Male</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Female</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>294</td>
      <td>Male</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <td>295</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <td>296</td>
      <td>Female</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <td>297</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <td>298</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>299 rows × 5 columns</p>
</div>




```python
sns.countplot(df['sex1'],data= df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2d54ee464c8>




![png](output_20_1.png)



```python
pd.options.display.float_format='{:,.2f}'.format
df.sex1.value_counts()/df.shape[0]
```




    Male     0.65
    Female   0.35
    Name: sex1, dtype: float64




```python
sns.countplot(df['high_blood_pressure'],data=df,hue=df['sex1'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2d54eef0c48>




![png](output_22_1.png)



```python
df.high_blood_pressure.value_counts()/df.shape[0]
```




    0   0.65
    1   0.35
    Name: high_blood_pressure, dtype: float64




```python
from collections import Counter
```


```python
Sur = ['high_blood_pressure','sex1']

Counter(df['high_blood_pressure']).most_common()
```




    [(0, 194), (1, 105)]




```python
for i in (df[Sur]):
    x = Counter(df[i])
    print(x)
```

    Counter({0: 194, 1: 105})
    Counter({'Male': 194, 'Female': 105})
    


```python
df.groupby(['high_blood_pressure','sex1']).agg(['count'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>age</th>
      <th>anaemia</th>
      <th>creatinine_phosphokinase</th>
      <th>diabetes</th>
      <th>ejection_fraction</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
      <th>sex</th>
      <th>smoking</th>
      <th>time</th>
      <th>DEATH_EVENT</th>
      <th>Marg</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
      <th>count</th>
    </tr>
    <tr>
      <th>high_blood_pressure</th>
      <th>sex1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" valign="top">0</td>
      <td>Female</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
    </tr>
    <tr>
      <td>Male</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
      <td>133</td>
    </tr>
    <tr>
      <td rowspan="2" valign="top">1</td>
      <td>Female</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
    </tr>
    <tr>
      <td>Male</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.pivot_table(values= ['age'], index=df[['high_blood_pressure','sex1']],columns=['diabetes','smoking'],margins=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="5" halign="left">age</th>
    </tr>
    <tr>
      <th></th>
      <th>diabetes</th>
      <th colspan="2" halign="left">0</th>
      <th colspan="2" halign="left">1</th>
      <th>All</th>
    </tr>
    <tr>
      <th></th>
      <th>smoking</th>
      <th>0</th>
      <th>1</th>
      <th>0</th>
      <th>1</th>
      <th></th>
    </tr>
    <tr>
      <th>high_blood_pressure</th>
      <th>sex1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" valign="top">0</td>
      <td>Female</td>
      <td>61.89</td>
      <td>nan</td>
      <td>56.41</td>
      <td>60.00</td>
      <td>58.98</td>
    </tr>
    <tr>
      <td>Male</td>
      <td>63.54</td>
      <td>59.13</td>
      <td>58.31</td>
      <td>61.56</td>
      <td>60.49</td>
    </tr>
    <tr>
      <td rowspan="2" valign="top">1</td>
      <td>Female</td>
      <td>60.15</td>
      <td>71.00</td>
      <td>61.13</td>
      <td>50.00</td>
      <td>60.88</td>
    </tr>
    <tr>
      <td>Male</td>
      <td>63.57</td>
      <td>64.24</td>
      <td>61.18</td>
      <td>64.00</td>
      <td>63.39</td>
    </tr>
    <tr>
      <td>All</td>
      <td></td>
      <td>62.49</td>
      <td>60.80</td>
      <td>58.62</td>
      <td>61.93</td>
      <td>60.83</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(df.corr(),linewidths=2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2d54ef6a888>




![png](output_29_1.png)



```python
##Z_SCORE

from scipy import stats

z_score=np.abs(stats.zscore(df.iloc[:,:-2]))
```


```python
print(np.where(z_score>3))
```

    (array([  1,   4,   9,  19,  28,  52,  52,  60,  64,  72, 103, 105, 109,
           131, 134, 171, 199, 217, 228, 296], dtype=int64), array([2, 8, 7, 8, 7, 2, 7, 2, 4, 2, 2, 6, 6, 7, 2, 2, 8, 7, 7, 6],
          dtype=int64))
    


```python
df_clean=df[(z_score<3.1).all(axis=1)]
```


```python
df_clean.shape
```




    (280, 15)




```python
df.shape
```




    (299, 15)




```python
x = df.sex.value_counts()

sns.countplot(y=df['diabetes'],order=x.index)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2d54f07e508>




![png](output_35_1.png)



```python
continous_var = ['age', 'creatinine_phosphokinase',
   'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

for i,col in enumerate(df_clean[continous_var]):
    fig= plt.figure(figsize=(15,15))
    plt.subplot(6,4,i*2+1)
    plt.grid(True)
    plt.title(col)
    sns.kdeplot(df_clean.loc[df_clean['Marg']=='No',col],label='alive',color='black',shade=True,
                kernel='gau',cut=0)
    sns.kdeplot(df_clean.loc[df_clean['Marg']=='Yes',col],label='died',color='red',shade=True,
                kernel='gau',cut=0)
    plt.subplot(6, 4, i*2+2)
    sns.boxplot(y = col, data = df_clean, x="Marg", palette = ["green", "red"])


```


![png](output_36_0.png)



![png](output_36_1.png)



![png](output_36_2.png)



![png](output_36_3.png)



![png](output_36_4.png)



![png](output_36_5.png)



```python
Ser=["creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"]
```


```python
df_clean.groupby('Marg')[Ser].agg([np.mean,np.median])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">creatinine_phosphokinase</th>
      <th colspan="2" halign="left">ejection_fraction</th>
      <th colspan="2" halign="left">platelets</th>
      <th colspan="2" halign="left">serum_creatinine</th>
      <th colspan="2" halign="left">serum_sodium</th>
      <th colspan="2" halign="left">time</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>Marg</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>No</td>
      <td>467.18</td>
      <td>231.00</td>
      <td>40.15</td>
      <td>38</td>
      <td>261,565.74</td>
      <td>262,500.00</td>
      <td>1.14</td>
      <td>1.00</td>
      <td>137.36</td>
      <td>137.00</td>
      <td>159.14</td>
      <td>174.00</td>
    </tr>
    <tr>
      <td>yes</td>
      <td>433.14</td>
      <td>249.50</td>
      <td>32.62</td>
      <td>30</td>
      <td>252,626.33</td>
      <td>254,500.00</td>
      <td>1.58</td>
      <td>1.30</td>
      <td>135.63</td>
      <td>135.50</td>
      <td>73.03</td>
      <td>47.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.kdeplot(data=df_clean['age'],data2=df_clean['platelets'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2d550f10548>




![png](output_39_1.png)



```python
df_clean.groupby(['sex1','high_blood_pressure','Marg']).size().unstack().fillna(0).apply(lambda x:x/x.sum(),axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marg</th>
      <th>No</th>
      <th>yes</th>
    </tr>
    <tr>
      <th>sex1</th>
      <th>high_blood_pressure</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" valign="top">Female</td>
      <td>0</td>
      <td>0.75</td>
      <td>0.25</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.64</td>
      <td>0.36</td>
    </tr>
    <tr>
      <td rowspan="2" valign="top">Male</td>
      <td>0</td>
      <td>0.70</td>
      <td>0.30</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.66</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_clean.groupby(['sex1','smoking','Marg']).size().unstack().fillna(0).apply(lambda x:x/x.sum(),axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marg</th>
      <th>No</th>
      <th>yes</th>
    </tr>
    <tr>
      <th>sex1</th>
      <th>smoking</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" valign="top">Female</td>
      <td>0</td>
      <td>0.71</td>
      <td>0.29</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.33</td>
      <td>0.67</td>
    </tr>
    <tr>
      <td rowspan="2" valign="top">Male</td>
      <td>0</td>
      <td>0.67</td>
      <td>0.33</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.71</td>
      <td>0.29</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 280 entries, 0 to 298
    Data columns (total 15 columns):
    age                         280 non-null float64
    anaemia                     280 non-null int64
    creatinine_phosphokinase    280 non-null int64
    diabetes                    280 non-null int64
    ejection_fraction           280 non-null int64
    high_blood_pressure         280 non-null int64
    platelets                   280 non-null float64
    serum_creatinine            280 non-null float64
    serum_sodium                280 non-null int64
    sex                         280 non-null int64
    smoking                     280 non-null int64
    time                        280 non-null int64
    DEATH_EVENT                 280 non-null int64
    sex1                        280 non-null object
    Marg                        280 non-null object
    dtypes: float64(3), int64(10), object(2)
    memory usage: 45.0+ KB
    


```python
var_binary = ['anaemia','diabetes','high_blood_pressure','smoking','sex']
```


```python
plt.figure(figsize=(12,6))
for i,col in enumerate(df_clean[var_binary]):
    plt.subplot(2,3,i+1)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(col,fontsize=13)
    plt.subplots_adjust(wspace=0.5,hspace=0.4)
    sns.countplot(x=col,data=df_clean,hue='Marg',palette=['red','blue'])
```


![png](output_44_0.png)



```python
df_clean.groupby(['Marg','sex1'])[var_binary].agg(np.mean)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>anaemia</th>
      <th>diabetes</th>
      <th>high_blood_pressure</th>
      <th>smoking</th>
      <th>sex</th>
    </tr>
    <tr>
      <th>Marg</th>
      <th>sex1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" valign="top">No</td>
      <td>Female</td>
      <td>0.47</td>
      <td>0.50</td>
      <td>0.40</td>
      <td>0.01</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>Male</td>
      <td>0.40</td>
      <td>0.37</td>
      <td>0.31</td>
      <td>0.48</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td rowspan="2" valign="top">yes</td>
      <td>Female</td>
      <td>0.55</td>
      <td>0.59</td>
      <td>0.52</td>
      <td>0.07</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>Male</td>
      <td>0.42</td>
      <td>0.33</td>
      <td>0.35</td>
      <td>0.44</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
##cross TABE

effect_sex_Marg = pd.crosstab(index=df_clean['sex1'],columns=df_clean['Marg'])
effect_sex_Marg.apply(lambda x:x/x.sum(),axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Marg</th>
      <th>No</th>
      <th>yes</th>
    </tr>
    <tr>
      <th>sex1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Female</td>
      <td>0.70</td>
      <td>0.30</td>
    </tr>
    <tr>
      <td>Male</td>
      <td>0.69</td>
      <td>0.31</td>
    </tr>
  </tbody>
</table>
</div>




```python
##cross TABE

effect_smoking_sex = pd.crosstab(index=df_clean['sex1'],columns=df_clean['smoking'])
effect_smoking_sex
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>smoking</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>sex1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Female</td>
      <td>94</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Male</td>
      <td>98</td>
      <td>85</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.lmplot(x='ejection_fraction',y= 'time',data=df_clean,hue='Marg')
```




    <seaborn.axisgrid.FacetGrid at 0x2d54fbb7e88>




![png](output_48_1.png)



```python
df.groupby('sex1')['ejection_fraction'].agg([np.mean,np.median])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>sex1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Female</td>
      <td>40.47</td>
      <td>38</td>
    </tr>
    <tr>
      <td>Male</td>
      <td>36.79</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>




```python
xx = pd.crosstab(index=df_clean['sex1'],columns=df['ejection_fraction'])

xx.apply(lambda z:z/z.sum())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>ejection_fraction</th>
      <th>14</th>
      <th>15</th>
      <th>17</th>
      <th>20</th>
      <th>25</th>
      <th>30</th>
      <th>35</th>
      <th>38</th>
      <th>40</th>
      <th>45</th>
      <th>50</th>
      <th>55</th>
      <th>60</th>
      <th>62</th>
      <th>65</th>
    </tr>
    <tr>
      <th>sex1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Female</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.12</td>
      <td>0.24</td>
      <td>0.41</td>
      <td>0.32</td>
      <td>0.39</td>
      <td>0.44</td>
      <td>0.21</td>
      <td>0.33</td>
      <td>1.00</td>
      <td>0.43</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <td>Male</td>
      <td>1.00</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.88</td>
      <td>0.76</td>
      <td>0.59</td>
      <td>0.68</td>
      <td>0.61</td>
      <td>0.56</td>
      <td>0.79</td>
      <td>0.67</td>
      <td>0.00</td>
      <td>0.57</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.lmplot(x='age',y='time',data=df_clean,hue='Marg',palette=['red','blue'])
```




    <seaborn.axisgrid.FacetGrid at 0x2d54fa04ac8>




![png](output_51_1.png)



```python
df_pre = df_clean.copy()
```


```python
df_pre.drop(columns=['sex1','Marg'],axis=1,inplace=True)
```


```python
from math import floor
```


```python
df_train = floor(df_pre.shape[0]*0.8)
df_test = floor(df_pre.shape[0]*0.2)
```


```python
df_train = df_pre.iloc[:244]
```


```python
df_test = df_pre.iloc[244:]
```


```python
df_train.columns
```




    Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
           'ejection_fraction', 'high_blood_pressure', 'platelets',
           'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
           'DEATH_EVENT'],
          dtype='object')




```python
y_train = df_train['DEATH_EVENT']
x_train = df_train.drop(columns=['DEATH_EVENT'])
```


```python
y_test = df_test['DEATH_EVENT']
x_test = df_test.drop(columns=['DEATH_EVENT'])
```


```python
from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(min_samples_split=13,max_depth=7,random_state=1)
```


```python
from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(min_samples_split=6,max_depth=3,random_state=21)
```


```python
dct_fit = dct.fit(x_train,y_train)

pred = dct_fit.predict(x_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix

cr = classification_report(y_test,pred)
cm = confusion_matrix(y_test,pred)


print('EVALATION_____')
print(cr)
print('\n')
print(cm)
```

    EVALATION_____
                  precision    recall  f1-score   support
    
               0       0.97      0.94      0.96        34
               1       0.33      0.50      0.40         2
    
        accuracy                           0.92        36
       macro avg       0.65      0.72      0.68        36
    weighted avg       0.93      0.92      0.92        36
    
    
    
    [[32  2]
     [ 1  1]]
    


```python
train_score  = dct.score(x_train,y_train)
test_score = dct.score(x_test,y_test)
print('The train score is {:.2f}'.format(train_score),'\n-----------')
print('The test score is {:.2f}'.format(test_score))

```

    The train score is 0.89 
    -----------
    The test score is 0.92
    


```python
dct.feature_importances_
```




    array([0.        , 0.        , 0.05082887, 0.        , 0.0409975 ,
           0.        , 0.        , 0.15265998, 0.09629405, 0.        ,
           0.        , 0.65921959])




```python
from sklearn.metrics import roc_auc_score
```


```python
roc_auc_score(y_test,pred)
```




    0.7205882352941175




```python
pred2 = dct_fit.predict(x_train)

roc_auc_score(y_train,pred2)
```




    0.8559523809523809




```python

```
