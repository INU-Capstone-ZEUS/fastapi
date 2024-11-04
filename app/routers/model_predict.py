from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


## Processing and Feature ============================================
def preprecessingData(df):
    ## ������ ��ó��
    # df['date'] = pd.to_datetime(df['��¥'].astype(str) + df['�ð�'].astype(str).str.zfill(4), format='%Y%m%d%H%M')
    # cols = list(df.columns)
    # cols.remove('��¥')
    # cols.remove('�ð�')
    # cols.remove('date')
    # df = df[["date"] + cols]
    # df['date'] = pd.to_datetime(df['date'])

    df["Amount"] = round(df["Amount"]/1000000, 4)

    return df


# ������ ��� ��� �Լ�
def calculatePriceBB(df, window=20, num_std=2):
    df['PriceBB_center'] = df['End'].rolling(window=window).mean()  # 20�� �̵� ���
    df['STD20'] = df['End'].rolling(window=window).std()   # 20�� ǥ�� ����
    df['PriceBB_upper'] = df['PriceBB_center'] + (num_std * df['STD20'])       # ��� ���
    df['PriceBB_lower'] = df['PriceBB_center'] - (num_std * df['STD20'])       # �ϴ� ���

    return df


# �̵�м� ��� �Լ�
def calculate_yellow_box(df):

    most_high = df.loc[df.index[0], "High"]
    most_low = df.loc[df.index[0], "Low"]
    yellow_line = []
    red_line = []

    for idx in df.index:

        if most_high < df.loc[idx, "High"]:
            most_high = df.loc[idx, "High"]

        if most_low > df.loc[idx, "Low"]:
            most_low = df.loc[idx, "Low"]

        avg = (most_high + most_low)/2
        yellow_line.append(avg)
        red_line.append((avg + most_high)/2)

    df["Yellow_line"] = yellow_line
    df["Red_line"] = red_line

    return df


# �ŷ���� ���� �׸���
def calculateTamountBB(df, window=20, num_std=2):
    df['AmountBB_center'] = df['Amount'].rolling(window=window).mean()  # 20�� �̵� ���
    df_std = df['Amount'].rolling(window=window).std()   # 20�� ǥ�� ����
    df['AmountBB_upper'] = df['AmountBB_center'] + (num_std * df_std)       # ��� ���
    df['AmountBB_lower'] = df['AmountBB_center'] - (num_std * df_std)       # �ϴ� ���
    df['AmountBB_lower'] = df['AmountBB_center'].apply(lambda x: 0 if x < 0 else x)

    return df




## Model ============================================
# LSTM �� ����
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, drop_out=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  #                        
        return out

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




## API ============================================
#           
router = APIRouter()

#   ?          
class ModelInput(BaseModel):
    # key-value ������ ��ųʸ��� ���´ٰ� ����.
    data: dict

@router.post("/predict")
async def predict(input_data: ModelInput):
    try:
        # PJH. 11.02
        # ��ųʸ��� ���� �����͸� ������ ���������� �����.
        # ������ ���� ����� �޶��� ��� (Ex. ��ųʸ��� �ƴ϶� ����°�� �� ���)
        # �� ModelInput�� ������ �����ؾ� �ϸ�, ���������� ���� �����͸�
        # df��� ���������������� �ϼ� ���Ѿ� �Ѵ�.
        df = pd.DataFrame(input_data.data)


        ## ��ó��
        df = preprecessingData(df)
        ## �Է� ��ó ����
        df = calculateTamountBB(df)
        df = calculate_yellow_box(df)
        df = calculatePriceBB(df)
        target_columns = ['End', 'Start', 'High', 'Low', 
                          'Amount', 'AmountBB_center', 'AmountBB_upper', 'AmountBB_lower',
                          'Red_line', 'Yellow_line',
                          'PriceBB_center', 'PriceBB_upper', 'PriceBB_lower']
        
        # scailing
        feature_columns = df[target_columns].values
        scaler = MinMaxScaler()
        
        ## �Է� ������ �ϼ� : ���� shape [1, 10, 13]
        X_windows = scaler.fit_transform(feature_columns)
        data_tensor = torch.tensor(X_windows, dtype=torch.float32).unsqueeze(0)[:, -10:, :]


        ## �߷� Start
        with torch.no_grad():
            output = model(data_tensor) # ���� ��� ������ : [1, 1]
            prediction = output.squeeze()
        
        preds = torch.round(torch.sigmoid(prediction))

        ## ��ȯ���� 0 Ȥ�� 1
        return {"prediction": preds.item()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))