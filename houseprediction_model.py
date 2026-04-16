import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
df=pd.read_csv("houseprice.csv")
print(df)
plt.scatter(df.area,df.price,color='red')
plt.xlabel('Area in (sq ft)')
plt.ylabel('Price in US(Dollar)')
# plt.show()
reg=linear_model.LinearRegression()
#reg.fit() means training the linear regression model
reg.fit(df[['area']],df.price)
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()
price=reg.predict([[3300]])
print("Predicated price:",price[0])
d=pd.read_csv("area.csv")
print(d)
predict=reg.predict(d)
print("Price:",predict)
d['prices']=predict
print(d)
d.to_csv("prediction.csv",index=False)
with open('house_model.pkl','wb') as f:
    pickle.dump(reg,f)
print("Model saved successfully as house_model.pkl")