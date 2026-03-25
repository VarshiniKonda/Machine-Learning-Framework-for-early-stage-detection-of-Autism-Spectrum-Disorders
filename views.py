from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
import pymysql
global username
accuracy = []
precision = []
recall = [] 
fscore = []
global toddler_encoder, children_encoder, adolescent_encoder, adults_encoder
global toddler_scaler, children_scaler, adolescent_scaler, adults_scaler
global toddler_vc, children_vc, adolescent_vc, adults_vc

#function to calculate all metrics
def calculateMetrics(algorithm, X_train, y_train, X_test, y_test):
    algorithm.fit(X_train, y_train)
    predict = algorithm.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    return algorithm

def processDataset(dataset_name, scaler, label_encoder, use_cols):
    dataset = pd.read_csv(dataset_name, usecols=use_cols)
    #applying dataset processing technique to convert non-numeric data to numeric data
    columns = dataset.columns
    types = dataset.dtypes.values
    for j in range(len(types)):
        name = types[j]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))#encode all str columns to numeric
            label_encoder.append([columns[j], le])
    dataset.fillna(0, inplace = True)#replace missing values

    #dataset shuffling & Normalization
    Y = dataset['ASD_traits'].ravel()
    dataset.drop(['ASD_traits'], axis = 1,inplace=True)
    X = dataset.values
    X = scaler.fit_transform(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffle dataset values
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    adaboost_cls = AdaBoostClassifier()
    adaboost_cls = calculateMetrics(adaboost_cls, X_train, y_train, X_test, y_test)

    rf_cls = RandomForestClassifier()
    rf_cls = calculateMetrics(rf_cls, X_train, y_train, X_test, y_test)

    dt_cls = DecisionTreeClassifier()
    dt_cls = calculateMetrics(dt_cls, X_train, y_train, X_test, y_test)

    knn_cls = KNeighborsClassifier(n_neighbors=3)
    knn_cls = calculateMetrics(knn_cls, X_train, y_train, X_test, y_test)

    nb_cls = GaussianNB()
    nb_cls = calculateMetrics(nb_cls, X_train, y_train, X_test, y_test)

    lr_cls = LogisticRegression()
    lr_cls = calculateMetrics(lr_cls, X_train, y_train, X_test, y_test)

    svm_cls = svm.SVC()
    svm_cls = calculateMetrics(svm_cls, X_train, y_train, X_test, y_test)

    lda_cls = LinearDiscriminantAnalysis()
    lda_cls = calculateMetrics(lda_cls, X_train, y_train, X_test, y_test)

    X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)
    estimators = [('ab', adaboost_cls), ('rf', rf_cls), ('dt', dt_cls), ('knn', knn_cls), ('nb', nb_cls), ('lr', lr_cls), ('svm', svm_cls), ('lda', lda_cls)]
    vc_cls = VotingClassifier(estimators = estimators)
    vc_cls = calculateMetrics(vc_cls, X_train, y_train, X_test, y_test)
    
    return adaboost_cls, rf_cls, dt_cls, knn_cls, nb_cls, lr_cls, svm_cls, lda_cls, vc_cls

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})
def TPredict(request):
    if request.method == 'GET':
        return render(request, 'TPredict.html', {})
def CPredict(request):
    if request.method == 'GET':
        return render(request, 'CPredict.html', {})
def AdoPredict(request):
    if request.method == 'GET':
        return render(request, 'AdoPredict.html', {})
def AduPredict(request):
    if request.method == 'GET':
        return render(request, 'AduPredict.html', {})
def Register(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})
def AdminLogin(request):
    if request.method == 'GET':
        return render(request, 'AdminLogin.html', {})

def RunScaling(request):
    if request.method == 'GET':        
        output= "<font size=3 color=blue>Features Scaling Process Completed</font>"
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)

def getMetrics(metric):
    ada_metric = metric[0], metric[9], metric[18], metric[27]
    rf_metric = metric[1], metric[10], metric[19], metric[28]
    dt_metric = metric[2], metric[11], metric[20], metric[29]
    knn_metric = metric[3], metric[12], metric[21], metric[30]
    nb_metric = metric[4], metric[13], metric[22], metric[31]
    lr_metric = metric[5], metric[14], metric[23], metric[32]
    svm_metric = metric[6], metric[15], metric[24], metric[33]
    lda_metric = metric[7], metric[16], metric[25], metric[34]
    vc_metric = metric[8], metric[17], metric[26], metric[35]
    return ada_metric, rf_metric, dt_metric, knn_metric, nb_metric, lr_metric, svm_metric, lda_metric, vc_metric

def getResults():
    ada_acc, rf_acc, dt_acc, knn_acc, nb_acc, lr_acc, svm_acc, lda_acc, vc_acc = getMetrics(accuracy)
    ada_pre, rf_pre, dt_pre, knn_pre, nb_pre, lr_pre, svm_pre, lda_pre, vc_pre = getMetrics(precision)
    ada_rec, rf_rec, dt_rec, knn_rec, nb_rec, lr_rec, svm_rec, lda_rec, vc_rec = getMetrics(recall)
    ada_f1, rf_f1, dt_f1, knn_f1, nb_f1, lr_f1, svm_f1, lda_f1, vc_f1 = getMetrics(fscore)
    names = ['Toddler', 'Childrens', 'Adolescent', 'Adults']
    plt.xlabel('Algorithms')
    plt.ylabel('Metrics')
    plt.grid(True)
    plt.plot(names, ada_acc, 'ro-', color="blue")
    plt.plot(names, rf_acc, 'ro-', color="red")
    plt.plot(names, dt_acc, 'ro-', color="brown")
    plt.plot(names, knn_acc, 'ro-', color="green")
    plt.plot(names, nb_acc, 'ro-', color="pink")
    plt.plot(names, lr_acc, 'ro-', color="orange")
    plt.plot(names, svm_acc, 'ro-', color="magenta")
    plt.plot(names, lda_acc, 'ro-', color="cyan")
    plt.plot(names, vc_acc, 'ro-', color="black")
    plt.legend(['AB', 'RF', 'DT', 'KNN', 'GNB', 'LR', 'SVM', 'LDA', 'VC'], loc='upper left')
    plt.title('All Algorithm Accuracy Graph')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.clf()
    plt.cla()
    algorithms = ['AdaBoost', 'Random Forest', 'Decision Tree', 'KNN', 'Naive Bayes', 'Logistic Regression', 'SVM', 'LDA', 'Voting Classifier']
    output = '<table border="1"><tr><th>Dataset Name</th><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>FSCORE</th></tr>'
    m = 0
    n = 0
    for i in range(len(accuracy)):
        output += '<tr><td><font size="3" color="black">'+names[m]+'</font></td>'
        output += '<td><font size="3" color="black">'+algorithms[n]+'</font></td>'
        output += '<td><font size="3" color="black">'+str(accuracy[i])+'</font></td>'
        output += '<td><font size="3" color="black">'+str(precision[i])+'</font></td>'
        output += '<td><font size="3" color="black">'+str(recall[i])+'</font></td>'
        output += '<td><font size="3" color="black">'+str(fscore[i])+'</font></td></tr>'
        n += 1
        if n == 9:
            n = 0
            m += 1
    output += '</table><br/>'
    return output, img_b64

def createDataFrame(patient_type, gender, age, jaundice, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10):
    data = []
    data.append([int(q1), int(q2), int(q3), int(q4), int(q5), int(q6), int(q7), int(q8), int(q9), int(q10), float(age), gender, jaundice])
    if patient_type == 'Toddler':
        cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Sex', 'Jaundice']
        encoder = toddler_encoder
        scaler = toddler_scaler
        classifier = toddler_vc
    if patient_type == 'Adolescent':
        cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'age', 'gender', 'jundice']
        encoder = adolescent_encoder
        scaler = adolescent_scaler
        classifier = adolescent_vc
    if patient_type == 'Children':
        cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10_Autism_Spectrum_Quotient', 'Age_Years', 'Sex', 'Jaundice']
        encoder = children_encoder
        scaler = children_scaler
        classifier = children_vc
    if patient_type == 'Adult':
        cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'age', 'gender', 'jundice']
        encoder = adults_encoder
        scaler = adults_scaler
        classifier = adults_vc
    data = pd.DataFrame(data, columns = cols)    
    for j in range(len(encoder)-1):
        le = encoder[j]
        data[le[0]] = pd.Series(le[1].transform(data[le[0]].astype(str)))#encode all str columns to numeric    
    data = data.values
    data = scaler.transform(data)
    predict = classifier.predict(data)[0]
    status = "<font size=3 color=green>No Autism Detected</font>"
    if predict == 1:
        status = "<font size=3 color=red>Autism Detected</font>"
    return status   

def PredictAction(request):
    if request.method == 'POST':
        global toddler_encoder, children_encoder, adolescent_encoder, adults_encoder
        global toddler_scaler, children_scaler, adolescent_scaler, adults_scaler
        global toddler_vc, children_vc, adolescent_vc, adults_vc

        patient_type = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        age = request.POST.get('t3', False)
        jaundice = request.POST.get('t4', False)
        q1 = request.POST.get('t5', False)
        q2 = request.POST.get('t6', False)
        q3 = request.POST.get('t7', False)
        q4 = request.POST.get('t8', False)
        q5 = request.POST.get('t9', False)
        q6 = request.POST.get('t10', False)
        q7 = request.POST.get('t11', False)
        q8 = request.POST.get('t12', False)
        q9 = request.POST.get('t13', False)
        q10 = request.POST.get('t14', False)
        if patient_type == 'Toddler' or patient_type == 'Adolescent' or patient_type == 'Adult':
            gender = gender.lower().strip()
            jaundice = jaundice.lower().strip()
        status = createDataFrame(patient_type, gender, age, jaundice, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10)    
        context= {'data':"Predicted Result = "+status}
        return render(request, 'UserScreen.html', context)

def RunML(request):
    if request.method == 'GET':
        global toddler_encoder, children_encoder, adolescent_encoder, adults_encoder
        global toddler_scaler, children_scaler, adolescent_scaler, adults_scaler
        global toddler_vc, children_vc, adolescent_vc, adults_vc
        global accuracy, precision, recall, fscore
        accuracy.clear()
        precision.clear()
        recall.clear()
        fscore.clear()
        toddler_encoder = []
        toddler_scaler = Normalizer()
        toddler_cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Sex', 'Jaundice', 'ASD_traits']
        toddler_adaboost, toddler_rf, toddler_dt, toddler_knn, toddler_nb, toddler_lr, toddler_svm, toddler_lda, toddler_vc = processDataset("Dataset/Toddler.csv", toddler_scaler,
                                                                                                                         toddler_encoder, toddler_cols)
        print("done")
        children_encoder = []
        children_scaler = Normalizer()
        children_cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10_Autism_Spectrum_Quotient', 'Age_Years', 'Sex', 'Jaundice', 'ASD_traits']
        children_adaboost, children_rf, children_dt, children_knn, children_nb, children_lr, children_svm, children_lda, children_vc = processDataset("Dataset/Children.csv",
                                                                                                                                 children_scaler, children_encoder,
                                                                                                                                 children_cols)
        print("done")
        adolescent_encoder = []
        adolescent_scaler = QuantileTransformer()
        adolescent_cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'age', 'gender', 'jundice', 'ASD_traits']
        adolescent_adaboost, adolescent_rf, adolescent_dt, adolescent_knn, adolescent_nb, adolescent_lr, adolescent_svm, adolescent_lda, adolescent_vc = processDataset("Dataset/Adolescent.csv",
                                                                                                                                 adolescent_scaler, adolescent_encoder,
                                                                                                                                 adolescent_cols)
        print("done")
        adults_encoder = []
        adults_scaler = QuantileTransformer()
        adults_cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'age', 'gender', 'jundice', 'ASD_traits']
        adults_adaboost, adults_rf, adults_dt, adults_knn, adults_nb, adults_lr, adults_svm, adults_lda, adults_vc = processDataset("Dataset/Adults.csv",
                                                                                                                                 adults_scaler, adults_encoder,
                                                                                                                                 adults_cols)
        output, img_b64 = getResults()
        context= {'data':output, 'img': img_b64}
        return render(request, 'AdminScreen.html', context)

def LoadDataset(request):
    if request.method == 'GET':
        output='<table border=1 align=center width=100%><tr>'
        columns = ['Dataset Name', 'Number of Rows', 'Number of Columns']
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        toddler = pd.read_csv("Dataset/Toddler.csv")
        children = pd.read_csv("Dataset/Children.csv")
        adolescent = pd.read_csv("Dataset/Adolescent.csv")
        adults = pd.read_csv("Dataset/Adults.csv")
        columns = ['Toddler', 'Children', 'Adolescent', 'Adults']
        output += '<tr><td><font size="3" color="black">'+columns[0]+'</td><td><font size="3" color="black">'+str(toddler.shape[0])+'</td>'
        output += '<td><font size="3" color="black">'+str(toddler.shape[1])+'</td></tr>'
        output += '<tr><td><font size="3" color="black">'+columns[1]+'</td><td><font size="3" color="black">'+str(children.shape[0])+'</td>'
        output += '<td><font size="3" color="black">'+str(children.shape[1])+'</td></tr>'
        output += '<tr><td><font size="3" color="black">'+columns[2]+'</td><td><font size="3" color="black">'+str(adolescent.shape[0])+'</td>'
        output += '<td><font size="3" color="black">'+str(adolescent.shape[1])+'</td></tr>'
        output += '<tr><td><font size="3" color="black">'+columns[3]+'</td><td><font size="3" color="black">'+str(adults.shape[0])+'</td>'
        output += '<td><font size="3" color="black">'+str(adults.shape[1])+'</td></tr>'
        output+= "</table></br></br></br></br>"
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)

def AdminLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            context= {'data':'Welcome '+username}
            return render(request, "AdminScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'AdminScreen.html', context)

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        page = "UserLogin.html"
        status = "Invalid login"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'autism',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    status = "Welcome "+uname
                    page = "UserScreen.html"
                    break		
        context= {'data': status}
        return render(request, page, context)
def SignupAction(request):
    if request.method == 'POST':
        person = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'autism',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == person:
                    output = email+" Username already exists"
                    break
        if output == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'autism',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup VALUES('"+person+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = 'Signup Process Completed'
        context= {'data':output}
        return render(request, 'Register.html', context)

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})
def AdminScreen(request):
    if request.method == 'GET':
       return render(request, 'AdminScreen.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

