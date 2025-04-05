from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

class CropModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CropModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.layer(x)

app = FastAPI()
model = CropModel(input_size=7, output_size=22)
model.load_state_dict(torch.load("crop_recommendation_model.pth", map_location=torch.device('cpu')))
model.eval()

index_to_label = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
                  'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
                  'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
                  'coconut', 'cotton', 'jute', 'coffee']

class InputData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predict")
def predict(data: InputData):
    x = torch.tensor([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]], dtype=torch.float32)
    output = model(x)
    prediction = torch.argmax(output, dim=1).item()
    return {"crop": index_to_label[prediction]}