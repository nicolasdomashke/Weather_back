import requests
import datetime
import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
from PIL import Image
from torchvision import transforms
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
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) 
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

API_KEY = '15bb7791d0e66c74ab77adf25b1961a7'
device = "cuda" if torch.cuda.is_available() else "cpu"

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

model3 = torch.load("new.pth", map_location=torch.device('cpu'), weights_only=False)
model3.eval()

with open("encoder.pkl", "rb") as f:
    loaded_encoder = pickle.load(f)

app = FastAPI()

def weather_request(lat, lon):
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

def get_precipitation(lat, lon):
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
    pred_label = y.item()

    pred_label = loaded_encoder.inverse_transform([pred_label])[0]
    
    return pred_label

def get_image():
    url = "https://rtsp.me/embed/Yb8Th9Q5/"

    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        
        video_tag = soup.find("video", id="video")
        if video_tag:
            poster_url = video_tag.get("poster")
            print("Poster URL:", poster_url)
            
            image_response = requests.get(poster_url)
            if image_response.status_code == 200:
                with open("poster_image.jpg", "wb") as f:
                    f.write(image_response.content)
                print("Изображение сохранено как poster_image.jpg")
            else:
                print("Ошибка при загрузке изображения.")
        else:
            print("Тег <video> не найден.")
    else:
        print("Ошибка при получении страницы:", response.status_code)

def recog_image():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    img_path = "poster_image.jpg" 
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)

        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    classes = ['cloudy', 'foggy', 'rainy', 'storm', 'shine']

    return classes[predicted_class]

@app.get("/prediction")
async def gen_pred(lat: float, lon: float):
    historical_data = weather_request(lat, lon)
    if historical_data:
        pred = weather_pred(historical_data)
        #precipitation = get_precipitation(lat, lon)
        #historical_data.extend(pred)
        #min_temp = float("inf")
        #max_temp = -float("inf")
        #wind_avr = 0
        #for i in range(16):
        #    min_temp = min(min_temp, historical_data[i][2])
        #    max_temp = max(max_temp, historical_data[i][2])
        #    wind_avr += historical_data[i][4]
        #wind_avr /= 16
        #weather_type = weather_class([(precipitation - M2[0]) / S2[0], (max_temp - M2[1]) / S2[1], (min_temp + 273 - M2[2]) / S2[2], (wind_avr - M2[3]) / S2[3]])
        get_image()
        weather_type = recog_image()
        return {"status": "success", "data": pred, "type": weather_type}
    else:
        return {"status": "error"} 

@app.get("/test")
async def test_reply():
    return {"status": "success"}