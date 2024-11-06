from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.data_processing import preprecessingData, calculatePriceBB, calculate_yellow_box, calculateTamountBB
from model.non_transformer import Model


## Model ============================================
class Args():
    def __init__(self, ):
        # self.model_name = model_name
        # self.loss_func = loss_func
        # self.target_time = target_time
        # self.target_feature = target_feature

        self.task_name = 'classification'
        self.seq_len = 10
        self.pred_len = 0
        self.label_len = 1
        self.moving_avg = 3
        self.num_class = 1
        # 예측할 피처 수
        self.c_out = 13
        
        ## 임베딩 관련 인자
        # 인코더, 디코더 입력 피처 수
        self.enc_in = 13
        self.dec_in = 13
        # 임베딩 및 레이어의 은닉 상태 크기
        self.d_model = 4
        # 임베딩 유형
        self.embed = 'fixed'
        # 입력 데이터 주기
        self.freq = 't'
        self.dropout = 0.3
        
        ## 어텐션, 인코더/디코더
        self.factor = 1
        self.output_attention = False
        # 어텐션 헤드 수
        self.n_heads = 4
        # 인코더, 디코더 레이어 수
        self.e_layers = 3
        self.d_layers = 1
        # 피드포워드 네트워크 차원. 주로 d_modl의 4배
        self.d_ff = 4 * self.n_heads
        # 활성화 함수. relu 혹은 gelu
        self.activation = 'gelu'

        # Projector 히든 레이어 차원 및 레이어 수
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2

        self.top_k = 5
        self.num_kernels = 4

        self.time_feature = False

args = Args()
model = Model(args)
model.load_state_dict(torch.load('./checkpoints/checkpoint_nontft.pth', weights_only=True))
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
        x_mark_enc = data_tensor[:, :, -4:]


        ## 추론 Start
        with torch.no_grad():
            output = model(data_tensor, x_mark_enc) # 예상 출력 사이즈 : [1, 1]
            prediction = output.squeeze()
        
        preds = torch.round(torch.sigmoid(prediction))

        ## 반환값은 0 혹은 1
        return {"prediction": preds.item()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
