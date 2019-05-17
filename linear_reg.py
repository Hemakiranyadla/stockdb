import mysql.connector
import pandas as pd
from pandas import set_option
from matplotlib import pyplot
import numpy
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
mySQLconnection = mysql.connector.connect(user='foouser',password='F88Pa%%**',host='134.209.144.239',database='stocksdb')
sql_select_Query = "select * from interview"
print(mySQLconnection)
cursor = mySQLconnection.cursor()
cursor.execute(sql_select_Query)
records = cursor.fetchall()

#closing database connection.
if(mySQLconnection .is_connected()):
    cursor.close()
    mySQLconnection.close()
    print("MySQL connection is closed")
print("Total number of rows in stocksdb  is - ", cursor.rowcount)
data = pd.DataFrame(records) 
print (data.shape)
print(data.head())

#To know about the correlation between the attributes 
from pandas import set_option
pd.set_option('display.width',None)
pd.set_option('precision',2)
correlation=data.corr(method='pearson')
print(correlation)
fig=pyplot.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlation,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=numpy.arange(1,6,1)
pyplot.show()


array = data.values
share_point = array[:,3:7]
share_value = array[:,7:8]

test_size=0.60
seed =5
X_train,X_test,Y_train,Y_test=train_test_split(share_point,share_value,test_size=test_size,random_state=seed)
model=LinearRegression()
model.fit(X_train,Y_train)
predict=model.score(X_test,Y_test)
print("accuracy: %3f%%"%(predict*100))

