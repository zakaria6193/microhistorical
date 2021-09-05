from flask import Flask, request,jsonify, session
import datetime as datedatedate
from datetime import datetime

from xgboost import XGBRegressor
import xgboost
import sklearn

from google_trans_new import google_translator






from flask_cors import CORS


import numpy as np
import pandas as pd

import pickle






import dateutil


import pyrebase
import numpy as np


app = Flask(__name__)


CORS(app)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
config = {"apiKey": "AIzaSyCQlJNxtcp_ambo4mGWH9LxhK6Wsr3VlSM",
          "authDomain": "projectbase-1fca0.firebaseapp.com",
          "databaseURL": "https://projectbase-1fca0-default-rtdb.europe-west1.firebasedatabase.app",
          "projectId": "projectbase-1fca0",
          "storageBucket": "projectbase-1fca0.appspot.com",
          "messagingSenderId": "821113244030",
          "appId": "1:821113244030:web:1f86f63dfbba3d08c4cb2f",
          "measurementId": "G-J76JKQ1XX5"}
firebase = pyrebase.initialize_app(config)
db = firebase.database()
auth=firebase.auth()




def handledate(date):
    date=str(date).split(' ')[0]
    return date








#fonction qui fixe la forme de la date from "Jan 26,2020" to "datetime")
def fixdate(date):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    first = date.split(' ')
    for i in range(len(months)):
        if first[0] == months[i]:
            first[0] = str(i + 1)
    first[1] = first[1].replace(',', '')
    date = [int(first[2]), int(first[0].replace(' ', '')), int(first[1])]
    return datetime(date[0], date[1], date[2])


#fonction qui transforme une serie temporelle vers une data supervisé ( x et y )
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values




def comparedate(new):
    date=new.split(' : ')[1]
    date=date.split('/')
    date=datedatedate.date(int('20'+date[2]),int(date[1]),int(date[0]))
    a_month = dateutil.relativedelta.relativedelta(months=4)
    return datedatedate.date.today() - a_month > date
@app.route('/')
def hello_world():
    return 'Hello World!'
@app.route('/realhistorical', methods=['POST', 'GET'])
def realhistorical():
    if request.method == 'GET':
        symbol = request.args.get('symbol').upper()
        dic = db.child('realhistorical').child(symbol).get().val()

        df = pd.DataFrame()
        for key in dic.keys():
            df[key] = dic[key]
        dfpred=df

        # In[3]:

        df = df.drop(columns=['change', 'high', 'low', 'open', 'vol'])

        # In[4]:

        def fixdate(date):
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            first = date.split(' ')
            for i in range(len(months)):
                if first[0] == months[i]:
                    first[0] = str(i + 1)
            first[1] = first[1].replace(',', '')
            date = [int(first[2]), int(first[0].replace(' ', '')), int(first[1])]
            return datetime(date[0], date[1], date[2])

        # In[5]:

        df['newdate'] = df['date'].apply(fixdate)
        df['price'] = df['price'].apply(float)

        #------------------------moving average---------------
        dfprice = pd.DataFrame(df.price)
        dfprice['monthly MA'] = dfprice.price.rolling(30).mean()
        dfprice['three MA'] = dfprice.price.rolling(90).mean()
        dfprice['date']=df['newdate']
        dfprice['monthly MA']= dfprice['monthly MA'].apply(str)
        dfprice['three MA'] = dfprice['three MA'].apply(str)
        dic['monthlyMA']=[]
        dic['threeMA'] = []
        for e in dfprice['monthly MA']:
            dic['monthlyMA'].append(e)
        for e in dfprice['three MA']:
            dic['threeMA'].append(e)

        # ------------------------profil---------------
        dicprofil=db.child('profil').get().val()
        dfprofil=pd.DataFrame()
        for k in dicprofil.keys():
            dfprofil[k]=dicprofil[k]
        industry = dfprofil['industry'][dfprofil['symbol'] == symbol].values[0]
        sector = dfprofil['sector'][dfprofil['symbol'] == symbol].values[0]
        story = dfprofil['story'][dfprofil['symbol'] == symbol].values[0]
        dic['industry'] = []
        for i in range(len(dic['price'])):
            dic['industry'].append(str(industry))
        dic['sector'] = []
        for i in range(len(dic['price'])):
            dic['sector'].append(str(sector))
        dic['story'] = []
        for i in range(len(dic['price'])):
            dic['story'].append(str(story))




        #-----------------------------------------------------

        df.set_index('newdate', inplace=True)
        del (df['date'])
        df.sort_index(inplace=True)
        dfpred['price'] = dfpred['price'].apply(float)

        lst = []
        for l in dfpred['price']:
            lst.append(l)

        dfpred['price'] = lst
        lst2 = []
        for l in dfpred['date']:
            lst2.append(l)

        dfpred['date'] = lst2
        dfpred['date'] = dfpred['date'].apply(fixdate)
        dfpred = dfpred.set_index('date')
        dfpred['price'] = dfpred['price'].apply(float)
        dfpred = dfpred.drop(columns=['change', 'high', 'low', 'open', 'vol'])
        data = series_to_supervised(dfpred[-20:].values, n_in=19)

        # split dataset

        # seed history with training dataset
        history = [x for x in data]

        train = np.asarray(history)
        print(train)
        file_name = symbol + ".pkl"


        open_file = open('stockmodelsxgboost/' + file_name, "rb")
        model = pickle.load(open_file)
        open_file.close()




        prediction=str(model.predict(train)[0])

        """
        # In[6]:

        dp = df.loc['2020-5-5':str(df.index[-1])]

        # In[7]:

        # In[8]:

        dp.index = pd.DatetimeIndex(dp.index).to_period('D')

        # In[9]:

        model = sm.tsa.statespace.SARIMAX(dp['price'], order=(0, 1, 0), seasonal_order=(1, 1, 1, 7))
        results = model.fit()

        # In[10]:

        # In[11]:

        # In[12]:

        newdate = dp.index[-1] + timedelta(days=1)
        forecast = results.forecast()[0]

        dic['forecast'] = []
        for i in range(len(dic['price'])):
            dic['forecast'].append(str(forecast))
        """


        dic['forecast'] = []
        for i in range(len(dic['price'])):
            dic['forecast'].append(str(prediction))

        c = db.child('news').child(symbol).get().val()

        if str(type(c)) != "<class 'NoneType'>":
            # translator = google_translator()
            # sentiment = tf.keras.models.load_model('sentiment/classifier.h5')



            dic['news']=[]
            # dic['translated'] = []
            # dic['sentiment']=[]
            for e in c:
                dic['news'].append(e)
            #     try:
            #         dic['translated'].append(translator.translate(e, lang_tgt='en'))
            #     except:
            #         dddx=546
            # for e in dic['translated']:
            #     X = tokenizer.texts_to_sequences([e])
            #     X = pad_sequences(X)
            #     model.predict(X)


        df['price']=df['price'].apply(str)
        dic['price']=[]
        for e in df['price']:
            dic['price'].append(e)

        return dic

if __name__ == '__main__':
    app.run()
