from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df=pd.read_csv("hotel_bookings.csv")
df.drop(['agent','company','country'],inplace=True, axis=1)
df['children'].fillna(0,inplace=True)

cat=df.select_dtypes(include='object').columns
#使用Label Encoder对非数字项进行编码
encode = LabelEncoder()
df['arrival_date_month'] = encode.fit_transform(df['arrival_date_month'])
df['meal'] = encode.fit_transform(df['meal'])
df['market_segment'] = encode.fit_transform(df['market_segment'])
df['distribution_channel'] = encode.fit_transform(df['distribution_channel'])
df['reserved_room_type'] = encode.fit_transform(df['reserved_room_type'])
df['assigned_room_type'] = encode.fit_transform(df['assigned_room_type'])
df['deposit_type'] = encode.fit_transform(df['deposit_type'])
df['customer_type'] = encode.fit_transform(df['customer_type'])
df['reservation_status'] = encode.fit_transform(df['reservation_status'])

#使用map函数将year转换为编码值
df['arrival_date_year'] = df['arrival_date_year'].map({2015:1, 2016:2, 2017:3})

#缩小lead_time和adr的映射范围
scaler = MinMaxScaler()
df['lead_time'] = scaler.fit_transform(df['lead_time'].values.reshape(-1,1))
df['adr'] = scaler.fit_transform(df['adr'].values.reshape(-1,1))

#选取与is_canceled相关性较强的属性
data = df[['reservation_status','total_of_special_requests','required_car_parking_spaces',
           'deposit_type','booking_changes','assigned_room_type','previous_cancellations',
           'distribution_channel','lead_time','is_canceled']]

#划分数据集
X = data.drop(['is_canceled'],axis= 1)
y = data['is_canceled']
#Logistics回归
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 2)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
#准确率
accuracy = logreg.score(X_test,y_test)
print(accuracy)
#混淆矩阵
matrix = confusion_matrix(y_test, y_pred.round())
df.loc[y_test.index]
result=pd.DataFrame()
result['Hotel Name']=df.loc[y_test.index].hotel
result['Booking_Possibility']=y_pred
#将结果存储为csv文件
result.to_csv('pred.csv')



