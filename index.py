import pandas as pd
import numpy as np
import plotly.express as px

data= pd.read_csv("deliverytime.txt")

print("Helo")
print(data.head())

#now looking at column insights
print(data.info())

print("\n")
print("Any Null Values")
#check if dataset contains any null values
print(data.isnull().sum())

#Calculating Distance between two latittudes and Longitudes
#using the haversine formula
# Set the earth's radius (in kilometers)
R = 6371

# Convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi/180)

# Function to calculate the distance between two points using the haversine formula
def distcalculate(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2-lat1)
    d_lon = deg_to_rad(lon2-lon1)
    a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
  
# Calculate the distance between each pair of points
data['distance'] = np.nan

for i in range(len(data)):
    data.loc[i, 'distance'] = distcalculate(data.loc[i, 'Restaurant_latitude'], 
                                        data.loc[i, 'Restaurant_longitude'], 
                                        data.loc[i, 'Delivery_location_latitude'], 
                                        data.loc[i, 'Delivery_location_longitude'])

print(data.head())

#Data Exploration
#exploring the data to find relationshipes between the features
figure = px.scatter(data_frame= data,
                    x="distance",
                    y="Time_taken(min)",
                    size="Time_taken(min)",
                    trendline="ols",
                    title="Relationship Between Distance and Time Taken")

figure.show()
#Linear relationship with delviery time and age of the deliverer
figure =px.scatter(data_frame =data,
                    x="Delivery_person_Age",
                    y="Time_taken(min)",
                    size="Time_taken(min)",
                    color="distance",
                    trendline="ols",
                    title="Relationship Between Time Taken and Age")

figure.show()

#linear relationship between tim taken to the delivery partner

figure = px.scatter(data_frame = data, 
                    x="Delivery_person_Ratings",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "distance",
                    trendline="ols", 
                    title = "Relationship Between Time Taken and Ratings")
figure.show()

#Looking into the food ordered by customer and
# and the type of vehicle used by the partner
# to see if it affects delivery time

fig = px.box(data, x="Type_of_vehicle",
                    y="Time_taken(min)",
                    color="Type_of_order",
                    title="Realtionship between type of food and order")
fig.show()

#from the data in the figures
#the features that affect food delivery time are
#age of the elivery timer
#ratings of the delivery partner
#distance between the restaurant and the delivery location

#Creating the time prediction model

#splitting data
from sklearn.model_selection import train_test_split
x = np.array(data[["Delivery_person_Age", 
                   "Delivery_person_Ratings", 
                   "distance"]])
y = np.array(data[["Time_taken(min)"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)

# creating the LSTM neural network model
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()


import pickle as rick
import os.path as pth

#check if model is saved
olamodel = model
file_path = r'ricky.pkl'
flag = pth.exists(file_path)
# training the model
if flag:  
    olamodel = rick.load(open('ricky.pkl','rb'))  
    print("we got model")
else:
    print("No Model")
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(xtrain, ytrain, batch_size=1, epochs=9)
    #saving model if doesnt exist else open
    rick.dump(model,open('ricky.pkl','wb'))
    olamodel = model




print("\n")
#Now test the Model
print("Food Delivery Time Prediction")
a = int(input("Age of Delivery Partner: "))
b = float(input("Ratings of Previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[a, b, c]])
print("Predicted Delivery Time in Minutes = ", olamodel.predict(features))