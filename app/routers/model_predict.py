from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, drop_out=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 시점의 출력을 사용
        return out

# 모델 초기화 및 로드
input_size = 13  
hidden_size = 64
output_size = 1
num_layers = 2
drop_out = 0.3

model = LSTMModel(input_size, hidden_size, output_size, num_layers, drop_out=drop_out)
#model.load_state_dict(torch.load('./checkpoint.pth'))
#model.load_state_dict(torch.load('routers/checkpoint.pth', weights_only=True))
model.load_state_dict(torch.load('/code/app/routers/checkpoint.pth', weights_only=True))
model.eval()

# 라우터 설정
router = APIRouter()

# 요청 데이터 모델
class ModelInput(BaseModel):
    data: list  # 3D 데이터를 평탄화한 리스트로 받습니다. e.g., [배치*시퀀스*입력 크기]

@router.post("/predict")
async def predict(input_data: ModelInput):
    try:
        # 입력 데이터 로드 및 형태 변환
        data = np.array(input_data.data).reshape(-1, 10, input_size)  # 배치, 시퀀스 길이, 입력 크기
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        # 예측 수행
        with torch.no_grad():
            output = model(data_tensor)
            prediction = output.numpy().tolist()

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
