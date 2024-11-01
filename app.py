import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

diabetes = pd.read_csv("diabetes.csv")
diabetes_mean_df=diabetes.groupby('Outcome').mean()

diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']] = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.nan)

#median
def median_target(var):
    temp = diabetes[diabetes[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

columns = diabetes.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    diabetes.loc[(diabetes['Outcome'] == 0 ) & (diabetes[i].isnull()), i] = median_target(i)[i][0]
    diabetes.loc[(diabetes['Outcome'] == 1 ) & (diabetes[i].isnull()), i] = median_target(i)[i][1]

# Outlier Detection
# IQR+Q1
# 50%
# 24.65->25%+50%
# 24.65->25%
for feature in diabetes:
    Q1 = diabetes[feature].quantile(0.25)
    Q3 = diabetes[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1-1.5*IQR
    upper = Q3+1.5*IQR
    if diabetes[(diabetes[feature]>upper)].any(axis=None):
        print(feature, "yes")
    else:
        print(feature, "no")
Q1 = diabetes.Insulin.quantile(0.25)
Q3 = diabetes.Insulin.quantile(0.75)
IQR = Q3-Q1
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR
diabetes.loc[diabetes['Insulin']>upper, "Insulin"] = upper

# LOF
# local outlier factor
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=10)
lof.fit_predict(diabetes)

diabetes_scores = lof.negative_outlier_factor_
# np.sort(diabetes_scores)[0:20]
thresold = np.sort(diabetes_scores)[7]

outlier = diabetes_scores>thresold
diabetes = diabetes[outlier]


X = diabetes.drop(columns='Outcome',axis=1)
y = diabetes['Outcome']

scalr = StandardScaler()
scalr.fit(X)
starded_data = scalr.fit_transform(X)
X = starded_data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=2)
svc = svm.SVC(kernel='linear')
svc.fit(X_train,y_train)
train_y_pred = svc.predict(X_train)
test_y_pred = svc.predict(X_test)

train_acc=print("train set accurac :", accuracy_score(y_train,train_y_pred))
test_acc=print("test set accurac :", accuracy_score(y_test,test_y_pred))

# create the Streamlit app
def app():

    img = Image.open(r"img.jpeg")
    img = img.resize((200,200))
    st.image(img,caption="Diabetes Image",width=200)


    st.title('Diabetes Prediction')

    # create the input form for the user to input new data
    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # make a prediction based on the user input
    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]

    np_arry_data = np.asarray(input_data)
    reshape_data = np_arry_data.reshape(1, -1)
    std_data = scalr.transform(reshape_data)
    prediction = svc.predict(std_data)

    # display the prediction to the user
    st.write('Based on the input features, the model predicts:')
    if (prediction[0] == 1):
        st.warning("this person has diabetes")
    else:
        st.success("this person has not diabetes")
    print(prediction)

    # display some summary statistics about the dataset
    st.header('Dataset Summary')
    st.write(diabetes.describe())

    st.header('Distribution by Outcome')
    st.write(diabetes_mean_df)

    # # display the model accuracy
    # st.header('Model Accuracy')
    # st.write(f'Train set accuracy: {train_acc:.2f}')
    # st.write(f'Test set accuracy: {test_acc:.2f}')


if __name__ == '__main__':
    app()