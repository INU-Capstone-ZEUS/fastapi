from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

# LSTM �� ����
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, drop_out=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # ������ ������ ����� ���
        return out

# �� �ʱ�ȭ �� �ε�
input_size = 13  
hidden_size = 64
output_size = 1
num_layers = 2
drop_out = 0.3

model = LSTMModel(input_size, hidden_size, output_size, num_layers, drop_out=drop_out)
#model.load_state_dict(torch.load('./checkpoint.pth'))
#model.load_state_dict(torch.load('routers/checkpoint.pth', weights_only=True))
model.load_state_dict(torch.load('./checkpoint.pth', weights_only=True))
model.eval()

# ����� ����
router = APIRouter()

# ��û ������ ��
class ModelInput(BaseModel):
    data: list  # 3D �����͸� ��źȭ�� ����Ʈ�� �޽��ϴ�. e.g., [��ġ*������*�Է� ũ��]

@router.post("/predict")
async def predict(input_data: ModelInput):
    try:
        # �Է� ������ �ε� �� ���� ��ȯ
        data = np.array(input_data.data).reshape(-1, 10, input_size)  # ��ġ, ������ ����, �Է� ũ��
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        # ���� ����
        with torch.no_grad():
            output = model(data_tensor)
            prediction = output.numpy().tolist()

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
