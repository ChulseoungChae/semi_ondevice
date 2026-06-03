import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import mixed_precision,  layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import tensorflow as tf
import time
import joblib
import sys

mixed_precision.set_global_policy('mixed_float16')

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# GPU 사용 확인
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 자동 할당 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 설정 완료!")
    except RuntimeError as e:
        print(e)
    
strategy = tf.distribute.MirroredStrategy()


# 학습 파라미터 설정
data_path = "../standard_TraceData_80"
window_size = 192  # 과거 n초 데이터 사용 (192초)
predict_steps = [10, 20, 30]  # n초 후 예측 (10, 20, 30 초후)
epochs_per_run = 200
batch_size = 256
#예측할 칼럼 리스트
predict_columns = [
    #'MFCMon_DCS',           ## MFC Dichlorosilane(DCS) 유량 모니터링 값
    'VG11',                 ## Baratron Gauge(의 압력 모니터링 값 (프로세스중 작용)
    'MFCMon_P.POS',         # MFC P.POS 위치 모니터링 값
    #'MFCMon_NH3',           ## MFC 암모니아(NH3) 유량 모니터링 값
    #'Step ID',
    #'MFCMon_F.PWR',
    'MFCMon_L.POS',         # MFC Left Position 위치 모니터링 값
    'VG12',                 # Baratron Gauge(의 압력 모니터링 값 (프로세스외 작용)
    'VG13',                 # Baratron Gauge(의 압력 모니터링 값 (프로세스외 작용)
    #'TempAct_C',            # 중앙 위치 실제 온도
    #'TempAct_U',            # 상부 위치 실제 온도
    #'TempAct_CU',           # 중앙 상부 위치 실제 온도
    #'TempAct_CL',           # 중앙 하부 위치 실제 온도
    #'TempAct_L',              
    'MFCMon_N2-1',          # MFC(Mass Flow Controller) N2-1 모니터링 값
    'MFCMon_N2-2',          # MFC N2-2 모니터링 값
    'MFCMon_N2-3',          # MFC N2-3 모니터링 값
    'MFCMon_N2-4',          # MFC N2-4 모니터링 값
    'APCValveMon'          # APC Valve 모니터링 값
    ]

temp_add_columns = [
    'TempSet_', #'Temp_Set_'
    'Power_HT.' #'Temp_HT_Power_'
]
temp_add_columns2 = [
    'Temp_Set_', #'Temp_Set_'
    'Temp_HT_Power_' #'Temp_HT_Power_'
]

step_reverse_dict = {'END': 2, 'STANDBY': 0, 'START': 1, 'B.UP': 17, 'WAIT': 3, 'S.P-1': 74, 'S.P-2': 75, 'R.UP1': 25, 'STAB1': 22, 'S.P-3': 76, 'M.P-3': 81, 'L.CHK': 72, 'PREPRG1': 44, 'EVAC1': 99, 'EVAC2': 100, 'N-EVA1': 111, 'CLOSE1': 128, 'SI-FL1': 119, 'SI-EVA1': 117, 'CHANGE': 152, 'N-PRE1': 113, 'N-FL1': 115, 'N-FL2': 116, 'pre-NH3P': 110, 'DEPO1': 49, 'post_NH3P': 135, 'N2PRG1': 103, 'SI-EVA4': 149, 'A.VAC2': 85, 'A.PRG2': 90, 'A.VAC1': 84, 'A.PRG1': 89, 'N2PRG2': 104, 'N2PRG3': 105, 'A.VAC3': 86, 'A.PRG3': 91, 'A.VAC4': 87, 'A.PRG4': 92, 'CYCLE1': 130, 'A.PRG5': 93, 'R.DOWN1': 31, 'B.FILL1': 94, 'B.FILL2': 95, 'B.FILL3': 96, 'B.FILL4': 97, 'B.FILL5': 98, 'B.DOWN': 18, 'None': 0, 'nan': 0, 'NaN': 0, 'null': 0, 'NULL': 0, 'IDLE': 0}

global_min = {'Step ID': 0, 'TempSet_U' : 180.0, 'TempSet_CU' : 180.0, 'TempSet_C' : 180.0, 'TempSet_CL' : 180.0, 'TempSet_L' : 180.0, 'Power_HT.U' : 0.0, 'Power_HT.CU' : 0.0, 'Power_HT.C' : 0.0, 'Power_HT.CL' : 0.0, 'Power_HT.L' : 0.0, 'TempAct_U': 180.0, 'TempAct_CU': 180.0, 'TempAct_C': 180.0, 'TempAct_CL': 180.0, 'TempAct_L': 180.0, 'VG13': 0.004, 'VG11': 0.7, 'VG12': 0.0061, 'APCValveMon': 0.0, 'ValveAct_2': 0, 'ValveAct_3': 0, 'ValveAct_4': 0, 'ValveAct_5': 0, 'ValveAct_9': 0, 'ValveAct_12': 0, 'ValveAct_14': 0, 'ValveAct_16': 0, 'ValveAct_26': 0, 'ValveAct_28': 0, 'ValveAct_29': 0, 'ValveAct_60': 0, 'ValveAct_63': 0, 'ValveAct_73': 0, 'ValveAct_80': 0, 'ValveAct_89': 0, 'ValveAct_90': 0, 'MFCMon_N2-1': -0.316, 'MFCMon_N2-2': 0.0, 'MFCMon_N2-3': -0.117, 'MFCMon_N2-4': -0.29, 'MFCMon_DCS': -0.754, 'MFCMon_NH3': 0.0, 'MFCMon_F2': 0.0, 'MFCMon_F.PWR': 0.0, 'MFCMon_L.POS': -5.566, 'MFCMon_P.POS': -5.566}
global_max = {'Step ID': 160, 'TempSet_U' : 677.8, 'TempSet_CU' : 677.8, 'TempSet_C' : 677.8, 'TempSet_CL' : 677.8, 'TempSet_L' : 677.8, 'Power_HT.U' : 55.0, 'Power_HT.CU' : 55.0, 'Power_HT.C' : 55.0, 'Power_HT.CL' : 55.0, 'Power_HT.L' : 55.0, 'TempAct_U': 677.8, 'TempAct_CU': 677.8, 'TempAct_C': 677.8, 'TempAct_CL': 677.8, 'TempAct_L': 677.8, 'VG13': 11.762, 'VG11': 771.6, 'VG12': 1.2467, 'APCValveMon': 100.0, 'ValveAct_2': 1, 'ValveAct_3': 1, 'ValveAct_4': 1, 'ValveAct_5': 1, 'ValveAct_9': 1, 'ValveAct_12': 1, 'ValveAct_14': 1, 'ValveAct_16': 1, 'ValveAct_26': 1, 'ValveAct_28': 1, 'ValveAct_29': 1, 'ValveAct_60': 1, 'ValveAct_63': 1, 'ValveAct_73': 1, 'ValveAct_80': 1, 'ValveAct_89': 1, 'ValveAct_90': 1, 'MFCMon_N2-1': 4.468, 'MFCMon_N2-2': 7.998, 'MFCMon_N2-3': 5.0, 'MFCMon_N2-4': 4.459, 'MFCMon_DCS': 1.6, 'MFCMon_NH3': 12.764, 'MFCMon_F2': 2.002, 'MFCMon_F.PWR': 0.398, 'MFCMon_L.POS': 48.633, 'MFCMon_P.POS': 150.0}

log_file = 'log.txt'
with open(log_file, 'w', encoding='utf-8') as f:
    f.write("")

def logg(content):
    print(content)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(content)
        f.write('\n')

# CSV 파일 리스트 가져오는 함수
def find_csv_files(base_dir):
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv") and '-checkpoint.csv' not in file:
                csv_files.append(os.path.join(root, file))
    csv_files.sort()
    return csv_files

def create_sequence(X, y, window, pred_steps):
    X_seqs, y_seqs = [], []
    max_steps = max(pred_steps)
    for i in range(len(X) - window - max_steps):
        X_seqs.append(X[i:i+window])
        y_seq = [y[i + window + h -1][0] for h in pred_steps]
        y_seqs.append(y_seq)
    return np.array(X_seqs), np.array(y_seqs)
        
# 공정 모니터링 변수
selected_cols = ['Step ID', 'MFCMon_N2-1', 'MFCMon_N2-2', 'MFCMon_N2-3', 'MFCMon_N2-4', 'MFCMon_F.PWR', 'MFCMon_L.POS', 'MFCMon_P.POS', 'MFCMon_DCS', 'MFCMon_NH3', 'MFCMon_F2', 'APCValveMon', 'VG11', 'VG12', 'VG13', 'TempAct_U', 'TempAct_CU', 'TempAct_C', 'TempAct_CL', 'TempAct_L', 'ValveAct_2', 'ValveAct_3', 'ValveAct_4', 'ValveAct_5', 'ValveAct_9', 'ValveAct_12', 'ValveAct_14', 'ValveAct_16', 'ValveAct_26', 'ValveAct_28', 'ValveAct_29', 'ValveAct_60', 'ValveAct_63', 'ValveAct_73', 'ValveAct_80', 'ValveAct_89', 'ValveAct_90']

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.d_model = d_model
        self.proj = None  # 초기화는 build에서 수행

    def build(self, input_shape):
        self.proj = layers.Dense(self.d_model)

    def call(self, x):
        # x: (batch_size, seq_len, num_features)
        batch_size = tf.shape(x)[0]
        seq_len = x.shape[1]
        num_features = x.shape[2]
        num_patches = seq_len // self.patch_len
        x = tf.reshape(x, [batch_size, num_patches, self.patch_len * num_features])
        return self.proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_len': self.patch_len,
            'd_model': self.d_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            patch_len=config.get('patch_len'),
            d_model=config.get('d_model'),
            **{k: v for k, v in config.items() if k not in ['patch_len', 'd_model', 'length']}  # <-- 'length' 제거
        )



class PositionalEncoding(layers.Layer):
    def __init__(self, length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=[1, self.length, self.d_model],
            initializer='random_normal'
        )

    def call(self, x):
        return x + self.pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            'length': self.length,
            'd_model': self.d_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            length=config.get('length'),
            d_model=config.get('d_model'),
            **{k: v for k, v in config.items() if k not in ['length', 'd_model', 'patch_len']}  # <-- 'patch_len' 제거
        )


def transformer_block(d_model, num_heads, ff_dim, dropout=0.1):
    inputs = layers.Input(shape=(None, d_model))
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = layers.Dropout(dropout)(x)
    x = x + inputs

    x2 = layers.LayerNormalization(epsilon=1e-6)(x)
    x2 = layers.Dense(ff_dim, activation='gelu')(x2)
    x2 = layers.Dense(d_model)(x2)
    x2 = layers.Dropout(dropout)(x2)
    outputs = x + x2
    return models.Model(inputs, outputs)

def build_patchtst_model(seq_len, patch_len, num_features, d_model=64,
                         num_heads=4, ff_dim=128, num_layers=2):
    inputs = layers.Input(shape=(seq_len, num_features))
    num_patches = seq_len // patch_len

    x = PatchEmbedding(patch_len, d_model)(inputs)
    x = PositionalEncoding(num_patches, d_model)(x)

    for _ in range(num_layers):
        x = transformer_block(d_model, num_heads, ff_dim)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='gelu')(x)  # <-- improved activation
    output = layers.Dense(3, dtype='float32')(x)  # Mixed precision에서 출력 float32 고정 필요
    return models.Model(inputs, output)

# 데이터 로드
print("\n\n============ 데이터 로드 =============")
csv_list = find_csv_files(data_path)
csv_list.sort()

print(f"Total data files : {len(csv_list)}")

# 학습 데이터
train_csv_list = csv_list[:-5]
print(f"Train data files : {len(train_csv_list)}")

# 테스트 데이터
test_csv_list = csv_list[-5:]
print(f"Test data files : {len(test_csv_list)}")


def get_adjusted_bounds(min_vals, max_vals, ratio=0.15, epsilon=1e-6):
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)
    ranges = max_vals - min_vals
    margin = ranges * ratio + epsilon
    min_adj = min_vals - margin
    max_adj = max_vals + margin
    return min_adj, max_adj


print(f"컬럼 개수 : {len(selected_cols)}")
print(selected_cols)

os.makedirs('./scaler_v5', exist_ok=True)
skip_s = True
if not os.path.isfile('./scaler_v5/scaler_X_main.pkl'):
    skip_s = False
for predict_column in predict_columns:
    if not os.path.isfile(f'./scaler_v5/scaler_y_{predict_column}.pkl'):
        skip_s = False
        break
if not skip_s:        
    # 스케일러 초기화
    scaler_X = MinMaxScaler()
    # 예측 컬럼별로 scaler 생성용 딕셔너리
    print("X, Y 스케일러 조정 중...")
    X_min = np.array([global_min[col] for col in selected_cols])
    X_max = np.array([global_max[col] for col in selected_cols])
    # X 스케일러 보정 및 fit
    X_min_adj, X_max_adj = get_adjusted_bounds(X_min, X_max, ratio=0.15)
    scaler_X.fit(np.vstack([X_min, X_max]))
    joblib.dump(scaler_X, './scaler_v5/scaler_X_main.pkl')
    print("입력 스케일러 저장 완료: ./scaler_v5/scaler_X_main.pkl")

    # Y 스케일러 보정 및 예측 컬럼별로 저장
    for col in predict_columns:
        y_min = np.array([global_min[col]])
        y_max = np.array([global_max[col]])
        y_min_adj, y_max_adj = get_adjusted_bounds(y_min, y_max, ratio=0.15)
        scaler_y = MinMaxScaler()
        scaler_y.fit(np.vstack([y_min, y_max]))
        filename = f"./scaler_v5/scaler_y_{col}.pkl"
        joblib.dump(scaler_y, filename)
        print(f"예측 스케일러 저장 완료: {filename}")
else:
    print("X, Y 스케일러 확인")

# 학습 시 loss weighting:
def get_weighted_mae(lval, hval, add_wight, loss_func_type):
    def loss(y_true, y_pred):
        weights = tf.where(tf.logical_and(y_true >= lval, y_true <= hval), add_wight, 1.0)  # 중심 정규화 기준
        if loss_func_type == 'mae': delta = tf.abs(y_true - y_pred)
        elif loss_func_type == 'mse': delta = tf.square(y_true - y_pred)
        return tf.reduce_mean(weights * delta)
    return loss
    


logg(f"\n\n=======================================================================================================")
logg(f"\n[{predict_steps}초 후] 예측 모델 학습 시작")

for predict_column in predict_columns:
    logg(f"\n\n=====================================================================")
    logg(f"\n[{predict_column}] 파라미터 학습 시작")
    scaler_y = joblib.load(f'./scaler_v5/scaler_y_{predict_column}.pkl')
    scaler_X = joblib.load('./scaler_v5/scaler_X_main.pkl')
    add_columns = []
    # EarlyStopping 설정
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',   # 모니터링할 지표
            factor=0.5,           # 줄일 비율 (기존 lr * 0.5)
            patience=4,           # 몇 epoch 동안 개선이 없을 때 줄일지
            min_lr=5e-6           # 최소 학습률 제한
        )
    # 이전 모델, 그래프, 메모리 초기화
    tf.keras.backend.clear_session()
    # 전략 내부 (strategy.scope) 안쪽에서 정의해야 함
    with strategy.scope():
        loss_func = 'mae'
        set_learning_rate=5e-4
        if predict_column == "VG11": 
            # 커스텀 weighted loss 함수 생성
            y_low, y_high = scaler_y.transform([[0]]), scaler_y.transform([[9]])
            loss_func = get_weighted_mae(y_low, y_high, 100.0, loss_func)
        elif 'Temp' in predict_column: 
            set_learning_rate=1e-3
            reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',   # 모니터링할 지표
                    factor=0.5,           # 줄일 비율 (기존 lr * 0.5)
                    patience=4,           # 몇 epoch 동안 개선이 없을 때 줄일지
                    min_lr=1e-4           # 최소 학습률 제한
                )
            temp_pos = predict_column.split('_')[-1]
            for add_col in temp_add_columns:
                add_columns.append(add_col+temp_pos)
            for add_col in temp_add_columns:
                add_columns.append(add_col+temp_pos)
            scaler_X = MinMaxScaler()
            # 예측 컬럼별로 scaler 생성용 딕셔너리
            logg("Temp X 스케일러 조정 중...")
            X_min = np.array([global_min[col] for col in selected_cols + add_columns])
            X_max = np.array([global_max[col] for col in selected_cols + add_columns])
            # X 스케일러 보정 및 fit
            X_min_adj, X_max_adj = get_adjusted_bounds(X_min, X_max, ratio=0.15)
            scaler_X.fit(np.vstack([X_min_adj, X_max_adj]))
            joblib.dump(scaler_X, f'./scaler_v5/scaler_X_{predict_column}.pkl')
        model = build_patchtst_model(
            seq_len=window_size,
            patch_len=16,
            num_features=len(selected_cols) + len(add_columns),  # input feature 수
            d_model=64,
            num_heads=4,
            ff_dim=128,
            num_layers=2
        )
                
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=set_learning_rate), loss=loss_func)
        
        # 데이터 로딩 및 전처리
        start_time = time.time()
        X_all, y_all = [], []
        cnt = 1
        for file in train_csv_list:
            logg(file)
            data = pd.read_csv(file, low_memory=False, usecols=selected_cols + add_columns + ['Step Name'])
            data = data[selected_cols + add_columns + ['Step Name']]
            step_ids = []
            for i in range(len(data)):
                step_ids.append(step_reverse_dict[str(data.iloc[i]['Step Name'])])
            data['Step ID'] = step_ids
            del data['Step Name']
            data = data.iloc[1::2].copy()  # 홀수 index
            data.dropna(inplace=True)
            #data =  data[(data['Step ID'] == 111) | (data['Step ID'] == 128) | (data['Step ID'] == 119) | (data['Step ID'] == 117) | (data['Step ID'] == 152) | (data['Step ID'] == 113) | (data['Step ID'] == 115) | (data['Step ID'] == 116)]
            data.reset_index(drop=True, inplace=True)
            X_data = scaler_X.transform(data.values)
            y_data = scaler_y.transform(data[[predict_column]].values)
            X_seq, y_seq = create_sequence(X_data, y_data, window_size, predict_steps)
            X_all.append(X_seq)
            y_all.append(y_seq)
            if cnt % 20 == 0 or cnt == len(train_csv_list):
                X_all = np.concatenate(X_all, axis=0)
                y_all = np.concatenate(y_all, axis=0)
                # train/val split
                val_split = 0.1
                split_idx = int(len(X_all) * (1 - val_split))
                X_train, X_val = X_all[:split_idx], X_all[split_idx:]
                y_train, y_val = y_all[:split_idx], y_all[split_idx:]
                X_train = X_train.astype(np.float32)
                y_train = y_train.astype(np.float32)
                X_val = X_val.astype(np.float32)
                y_val = y_val.astype(np.float32)
                

                # tf.data.Dataset으로 변환 후 배치 처리 (drop_remainder 적용)
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size, drop_remainder=True)
                val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size, drop_remainder=True)

                # 학습
                model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs_per_run,
                    batch_size=batch_size,
                    callbacks=[reduce_lr, early_stop],
                    verbose=0
                )
                X_all, y_all = [], []     
            cnt+=1
        os.makedirs('./patchtst_model_v5', exist_ok=True)
        # 모델 저장 경로
        model_path = f'patchtst_model_v5/{loss_func}_{window_size}_patchtst_{predict_column}_main'
        # ② SavedModel 포맷 (.keras 디렉토리 형식) 저장
        model.save(f'{model_path}.keras', save_format='keras')  # 또는 save_format='tf'
        logg(f"[✔] 모델 저장 완료: {model_path}.keras")
        end_time = time.time()
        running_time = int(end_time - start_time)
        logg(f"\ntotal running time : {running_time} secs")
