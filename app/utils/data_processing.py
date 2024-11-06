import pandas as pd
import numpy as np


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