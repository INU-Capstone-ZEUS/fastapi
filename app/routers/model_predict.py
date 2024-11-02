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
    df['date'] = pd.to_datetime(df['날짜'].astype(str) + df['시간'].astype(str).str.zfill(4), format='%Y%m%d%H%M')
    cols = list(df.columns)
    cols.remove('날짜')
    cols.remove('시간')
    cols.remove('date')
    df = df[["date"] + cols]
    df['date'] = pd.to_datetime(df['date'])

    df["거래대금"] = round(df["거래대금"]/1000000, 4)

    return df


# 볼린저 밴드 계산 함수
def calculatePriceBB(df, window=20, num_std=2):
    df['주가볼밴_중심선'] = df['종가'].rolling(window=window).mean()  # 20일 이동 평균
    df['STD20'] = df['종가'].rolling(window=window).std()   # 20일 표준 편차
    df['주가볼밴_상단선'] = df['주가볼밴_중심선'] + (num_std * df['STD20'])       # 상단 밴드
    df['주가볼밴_하단선'] = df['주가볼밴_중심선'] - (num_std * df['STD20'])       # 하단 밴드

    return df


# 이등분선 계산 함수
def calculate_yellow_box(df):

    most_high = df.loc[df.index[0], "고가"]
    most_low = df.loc[df.index[0], "저가"]
    yellow_line = []
    red_line = []

    for idx in df.index:

        if most_high < df.loc[idx, "고가"]:
            most_high = df.loc[idx, "고가"]

        if most_low > df.loc[idx, "저가"]:
            most_low = df.loc[idx, "저가"]

        avg = (most_high + most_low)/2
        yellow_line.append(avg)
        red_line.append((avg + most_high)/2)

    df["이등분선"] = yellow_line
    df["이등상단"] = red_line

    return df


# 거래대금 볼밴 그리기
def calculateTamountBB(df, window=20, num_std=2):
    df['거래대금_중심선'] = df['거래대금'].rolling(window=window).mean()  # 20일 이동 평균
    df_std = df['거래대금'].rolling(window=window).std()   # 20일 표준 편차
    df['거래대금_상단선'] = df['거래대금_중심선'] + (num_std * df_std)       # 상단 밴드
    df['거래대금_하단선'] = df['거래대금_중심선'] - (num_std * df_std)       # 하단 밴드
    df['거래대금_하단선'] = df['거래대금_하단선'].apply(lambda x: 0 if x < 0 else x)

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
        out = self.fc(out[:, -1, :])  # ������ ������ ����� ���
        return out

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




## API ============================================
# ����� ����
router = APIRouter()

# ��û ������ ��
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
        target_columns = ['종가', '시가', '고가', '저가', 
                          '거래대금', '거래대금_중심선', '거래대금_상단선', '거래대금_하단선',
                          '이등상단', '이등분선',
                          '주가볼밴_중심선', '주가볼밴_상단선', '주가볼밴_하단선']
        
        # scailing
        feature_columns = df[target_columns].values
        scaler = MinMaxScaler()
        
        ## 입력 시퀀스 완성 : 예상 shape [1, 10, 13]
        X_windows = scaler.fit_transform(feature_columns)
        data_tensor = torch.tensor(X_windows, dtype=torch.float32).unsqueeze(0) 


        ## 추론 Start
        with torch.no_grad():
            output = model(data_tensor) # 예상 출력 사이즈 : [1, 1]
            prediction = output.squeeze()
        
        preds = torch.round(torch.sigmoid(prediction))

        ## 반환값은 0 혹은 1
        return {"prediction": preds.item()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
