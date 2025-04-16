import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score


data = {
    'Day': ['sunny','windy','sunny','windy','windy','sunny','windy','sunny','sunny','windy'],
    'Temp': ['cool','hot','cool','hot','hot','hot','cool','hot','hot','hot'],
    'class': ['play','not play','not play','play','play','play','not play','play','play','play' ]
}

df = pd.DataFrame(data)
print(df)
X_raw = df[['Day','Temp']]
Y_raw = df['class']

Onehot_encoder = OneHotEncoder()
x_encoded = Onehot_encoder.fit_transform(X_raw).toarray()

label_Encoder = LabelEncoder()
y_encoded = label_Encoder.fit_transform(Y_raw)

x_train,x_test, y_train ,y_test = train_test_split(x_encoded,y_encoded,test_size=0.3,random_state=42)
model = GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("Enter the day and temperature to predict the class:")
day = input("Enter the day: ")
temp = input("Enter the temperature: ")

new_instance = pd.DataFrame([[day,temp]], columns=['Day','Temp'])
new_instance_encoded = Onehot_encoder.transform(new_instance).toarray()
predicted_class = model.predict(new_instance_encoded)
predicted_label = label_Encoder.inverse_transform(predicted_class)[0]
print("Prediction for Day = Windy,temp = cool",predicted_label)

