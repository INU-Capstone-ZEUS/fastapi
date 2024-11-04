from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


## Processing and Feature ============================================
def preprecessingData(df):
    ## 데이터 전처리
    # df['date'] = pd.to_datetime(df['날짜'].astype(str) + df['시간'].astype(str).str.zfill(4), format='%Y%m%d%H%M')
    # cols = list(df.columns)
    # cols.remove('날짜')
    # cols.remove('시간')
    # cols.remove('date')
    # df = df[["date"] + cols]
    # df['date'] = pd.to_datetime(df['date'])

    df["Amount"] = round(df["Amount"]/1000000, 4)

    return df


# 볼린저 밴드 계산 함수
def calculatePriceBB(df, window=20, num_std=2):
    df['PriceBB_center'] = df['End'].rolling(window=window).mean()  # 20일 이동 평균
    df['STD20'] = df['End'].rolling(window=window).std()   # 20일 표준 편차
    df['PriceBB_upper'] = df['PriceBB_center'] + (num_std * df['STD20'])       # 상단 밴드
    df['PriceBB_lower'] = df['PriceBB_center'] - (num_std * df['STD20'])       # 하단 밴드

    return df


# 이등분선 계산 함수
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


# 거래대금 볼밴 그리기
def calculateTamountBB(df, window=20, num_std=2):
    df['AmountBB_center'] = df['Amount'].rolling(window=window).mean()  # 20일 이동 평균
    df_std = df['Amount'].rolling(window=window).std()   # 20일 표준 편차
    df['AmountBB_upper'] = df['AmountBB_center'] + (num_std * df_std)       # 상단 밴드
    df['AmountBB_lower'] = df['AmountBB_center'] - (num_std * df_std)       # 하단 밴드
    df['AmountBB_lower'] = df['AmountBB_center'].apply(lambda x: 0 if x < 0 else x)

    return df




## Model ============================================
# LSTM 모델 선언
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
    # key-value 형태의 딕셔너리로 들어온다고 가정.
    data: dict

@router.post("/predict")
async def predict(input_data: ModelInput):
    try:
        # PJH. 11.02
        # 딕셔너리로 받은 데이터를 데이터 프레임으로 만들기.
        # 데이터 수신 방식이 달라질 경우 (Ex. 딕셔너리가 아니라 파일째로 올 경우)
        # 위 ModelInput의 형식을 수정해야 하먀, 최종적으로 받은 데이터를
        # df라는 데이터프레임으로 완성 시켜야 한다.
        df = pd.DataFrame(input_data.data)


        ## 전처리
        df = preprecessingData(df)
        ## 입력 피처 생성
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
        
        ## 입력 시퀀스 완성 : 예상 shape [1, 10, 13]
        X_windows = scaler.fit_transform(feature_columns)
        data_tensor = torch.tensor(X_windows, dtype=torch.float32).unsqueeze(0)[:, -10:, :]


        ## 추론 Start
        with torch.no_grad():
            output = model(data_tensor) # 예상 출력 사이즈 : [1, 1]
            prediction = output.squeeze()
        
        preds = torch.round(torch.sigmoid(prediction))

        ## 반환값은 0 혹은 1
        return {"prediction": preds.item()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
