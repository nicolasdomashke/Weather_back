import requests
import datetime
import torch
import torch.nn as nn
import pickle
import numpy as np
from fastapi import FastAPI
#from fastapi.responses import Response
#from pydantic import BaseModel

class WeatherPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(WeatherPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4 * input_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        fc_out = self.fc(last_hidden)
        fc_out = fc_out.view(-1, 4, x.size(2))
        return fc_out
    
class WeatherClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(WeatherClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

API_KEY = '15bb7791d0e66c74ab77adf25b1961a7'
device = "cuda" if torch.cuda.is_available() else "cpu"

lat = 55.7522
lon = 37.6156

with open("mean.pkl", "rb") as f:
    M = pickle.load(f)

with open("square.pkl", "rb") as f:
    S = pickle.load(f)

with open("mean2.pkl", "rb") as f:
    M2 = pickle.load(f)

with open("square2.pkl", "rb") as f:
    S2 = pickle.load(f)

model = WeatherPredictor(5, 50)
model.load_state_dict(torch.load("weather_predictor.pth", map_location=torch.device('cpu')))
model.eval() 
#model.to(device)

model2 = WeatherClassifier(4, 64, 5)
model2.load_state_dict(torch.load("weather_classifier.pth", map_location=torch.device('cpu')))
model2.eval() 
#model2.to(device)

app = FastAPI()

def weather_request():
    X = []
    for i in range(12):
        dt = int((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=i)).timestamp())

        url = (
            f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
            f"?lat={lat}&lon={lon}&dt={dt}&appid={API_KEY}&units=metric"
        )

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()["data"][0]
            
            vector = [(data["humidity"] - M[0]) / S[0], (data["pressure"] - M[1]) / S[1], (data["temp"] + 273 - M[2]) / S[2], (data["wind_deg"] - M[3]) / S[3], (data["wind_speed"] - M[4]) / S[4]]
            X.append(vector)
        else:
            print("Ошибка запроса:", response.status_code, response.text)
            return
    
    X.reverse()
    return X

def get_precipitation():
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    precipitation = 0
    if "rain" in data:
        precipitation = data["rain"].get("1h", 0)
    elif "snow" in data:
        precipitation = data["snow"].get("1h", 0)

    return precipitation

def weather_pred(X):
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    #X_tensor = X_tensor.to(device)
    with torch.no_grad():
        y = model(X_tensor)
    y = y.squeeze(0).cpu().numpy()
    y = [[vector[i] * S[i] + M[i] - (273 if i == 2 else 0) for i in range(5)] for vector in y]
    return y

def weather_class(X):
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    #X_tensor = X_tensor.to(device)
    with torch.no_grad():
        outputs = model2(X_tensor)
        _, y = torch.max(outputs, 1)
    return y

@app.get("/prediction")
async def gen_pred():
    historical_data = weather_request()
    if historical_data:
        pred = weather_pred(historical_data)
        precipitation = get_precipitation()
        historical_data.extend(pred)
        min_temp = float("inf")
        max_temp = -float("inf")
        wind_avr = 0
        for i in range(16):
            min_temp = min(min_temp, historical_data[i][2])
            max_temp = max(max_temp, historical_data[i][2])
            wind_avr += historical_data[i][4]
        wind_avr /= 16
        weather_type = weather_class([(precipitation - M2[0]) / S2[0], (max_temp - M2[1]) / S2[1], (min_temp + 273 - M2[2]) / S2[2], (wind_avr - M2[3]) / S2[3]])
        print([(precipitation - M2[0]) / S2[0], (max_temp - M2[1]) / S2[1], (min_temp + 273 - M2[2]) / S2[2], (wind_avr - M2[3]) / S2[3]])
        return {"status": "success", "data": pred, "type": weather_type}
    else:
        return {"status": "error"} 

@app.get("/test")
async def test_reply():
    return {"status": "success"}