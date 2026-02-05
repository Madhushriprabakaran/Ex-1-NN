<H3>ENTER YOUR NAME: MADHUSHRI</H3>
<H3>ENTER YOUR REGISTER NO:212224040178 </H3> 
<H3>EX. NO.1</H3>

<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


```


## OUTPUT:
## Dataset:
<img width="912" height="303" alt="image" src="https://github.com/user-attachments/assets/00c0ff76-7e26-410e-a132-9f09b5e79cfb" />

## X Values:

<img width="803" height="206" alt="image" src="https://github.com/user-attachments/assets/88401c0c-b8bc-4974-af11-6892ba1ff3ed" />

## Y Values:

<img width="497" height="47" alt="image" src="https://github.com/user-attachments/assets/ddee2558-100f-402d-8794-9ac93f59d484" />

## Null Values:

<img width="287" height="382" alt="image" src="https://github.com/user-attachments/assets/f255a010-cae2-4634-8f4d-0adb235876b5" />

## Duplicated Values:

<img width="405" height="312" alt="image" src="https://github.com/user-attachments/assets/303df82e-01e1-45c9-94df-8e9ca880cbf3" />

## Description:

<img width="898" height="231" alt="image" src="https://github.com/user-attachments/assets/274cce6e-e529-426c-ad28-eb3fba2a3cc8" />

## Normalized Dataset:

<img width="833" height="667" alt="image" src="https://github.com/user-attachments/assets/461abb53-11b2-4d77-8102-d6cf7fbe6654" />

## Training Data:

<img width="877" height="207" alt="image" src="https://github.com/user-attachments/assets/2f08e3ca-2467-4bd5-9a3e-c2f3b73ae180" />

## Testing Data:

<img width="828" height="247" alt="image" src="https://github.com/user-attachments/assets/6e79724d-d678-4c55-a17f-d5563bbe511e" />









## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


