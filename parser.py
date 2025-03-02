import requests
import datetime
import torch
import torch.nn as nn
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

API_KEY = '15bb7791d0e66c74ab77adf25b1961a7'

lat = 55.7522
lon = 37.6156

with open("mean.pkl", "rb") as f:
    M = pickle.load(f)

with open("square.pkl", "rb") as f:
    S = pickle.load(f)

model = WeatherPredictor(5, 50)
model.load_state_dict(torch.load("weather_predictor.pth"))
model.eval() 
model.to("cuda")

app = FastAPI()

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

def weather_pred(X):
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    X_tensor = X_tensor.to("cuda")
    with torch.no_grad():
        y = model(X_tensor)
    y = y.squeeze(0).cpu().numpy()
    y = [[vector[i] * S[i] + M[i] - (273 if i == 2 else 0) for i in range(5)] for vector in y]
    return y


@app.get("/prediction")
async def gen_pred():
    historical_data = weather_request()
    if historical_data:
        pred = weather_pred(historical_data)
        return {"status": "success", "data": pred}
    else:
        return {"status": "error"} 