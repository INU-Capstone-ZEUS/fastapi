from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .utils.data_processing import preprecessingData, calculatePriceBB, calculate_yellow_box, calculateTamountBB
from .model.non_transformer import Model


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
        # ������ ��ó ��
        self.c_out = 13
        
        ## �Ӻ��� ���� ����
        # ���ڴ�, ���ڴ� �Է� ��ó ��
        self.enc_in = 13
        self.dec_in = 13
        # �Ӻ��� �� ���̾��� ���� ���� ũ��
        self.d_model = 4
        # �Ӻ��� ����
        self.embed = 'fixed'
        # �Է� ������ �ֱ�
        self.freq = 't'
        self.dropout = 0.3
        
        ## ���ټ�, ���ڴ�/���ڴ�
        self.factor = 1
        self.output_attention = False
        # ���ټ� ��� ��
        self.n_heads = 4
        # ���ڴ�, ���ڴ� ���̾� ��
        self.e_layers = 3
        self.d_layers = 1
        # �ǵ������� ��Ʈ��ũ ����. �ַ� d_modl�� 4��
        self.d_ff = 4 * self.n_heads
        # Ȱ��ȭ �Լ�. relu Ȥ�� gelu
        self.activation = 'gelu'

        # Projector ���� ���̾� ���� �� ���̾� ��
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2

        self.top_k = 5
        self.num_kernels = 4

        self.time_feature = False

args = Args()
model = Model(args)
model.load_state_dict(torch.load('/code/app/routers/checkpoints/checkpoint_nontft.pth', weights_only=True))
model.eval()



## API ============================================
router = APIRouter()

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
        x_mark_enc = data_tensor[:, :, -4:]


        ## �߷� Start
        with torch.no_grad():
            output = model(data_tensor, x_mark_enc) # ���� ��� ������ : [1, 1]
            prediction = output.squeeze()
        
        preds = torch.round(torch.sigmoid(prediction))

        ## ��ȯ���� 0 Ȥ�� 1
        return {"prediction": preds.item()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))