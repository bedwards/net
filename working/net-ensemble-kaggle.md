```python
import warnings

warnings.simplefilter("ignore")

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import cudf

if torch.cuda.is_available():
    device = "cuda"
else:
    raise

kfold = KFold(shuffle=True, random_state=42)
```


```python
fn = "train_poss.csv"
print(f"reading {fn}")
train = pd.read_csv(f"../input/net-dataset/{fn}")

train = pd.concat(
    [
        train.select_dtypes("int64").astype("int32"),
        train.select_dtypes("float64").astype("float32"),
    ],
    axis=1,
)

print(f"\ntrain: {str(train.shape):>23}")
print(f"{train.columns.to_list()}")
```

    reading train_poss.csv
    
    train:           (202033, 117)
    ['Season', 'DayNum', 'TeamID_1', 'TeamID_2', 'Margin', 'Score_poss_o_1', 'FGM_poss_o_1', 'FGA_poss_o_1', 'FGM3_poss_o_1', 'FGA3_poss_o_1', 'FTM_poss_o_1', 'FTA_poss_o_1', 'OR_poss_o_1', 'DR_poss_o_1', 'Ast_poss_o_1', 'TO_poss_o_1', 'Stl_poss_o_1', 'Blk_poss_o_1', 'PF_poss_o_1', 'Score_poss_d_1', 'FGM_poss_d_1', 'FGA_poss_d_1', 'FGM3_poss_d_1', 'FGA3_poss_d_1', 'FTM_poss_d_1', 'FTA_poss_d_1', 'OR_poss_d_1', 'DR_poss_d_1', 'Ast_poss_d_1', 'TO_poss_d_1', 'Stl_poss_d_1', 'Blk_poss_d_1', 'PF_poss_d_1', 'sos_Score_poss_o_1', 'sos_FGM_poss_o_1', 'sos_FGA_poss_o_1', 'sos_FGM3_poss_o_1', 'sos_FGA3_poss_o_1', 'sos_FTM_poss_o_1', 'sos_FTA_poss_o_1', 'sos_OR_poss_o_1', 'sos_DR_poss_o_1', 'sos_Ast_poss_o_1', 'sos_TO_poss_o_1', 'sos_Stl_poss_o_1', 'sos_Blk_poss_o_1', 'sos_PF_poss_o_1', 'sos_Score_poss_d_1', 'sos_FGM_poss_d_1', 'sos_FGA_poss_d_1', 'sos_FGM3_poss_d_1', 'sos_FGA3_poss_d_1', 'sos_FTM_poss_d_1', 'sos_FTA_poss_d_1', 'sos_OR_poss_d_1', 'sos_DR_poss_d_1', 'sos_Ast_poss_d_1', 'sos_TO_poss_d_1', 'sos_Stl_poss_d_1', 'sos_Blk_poss_d_1', 'sos_PF_poss_d_1', 'Score_poss_o_2', 'FGM_poss_o_2', 'FGA_poss_o_2', 'FGM3_poss_o_2', 'FGA3_poss_o_2', 'FTM_poss_o_2', 'FTA_poss_o_2', 'OR_poss_o_2', 'DR_poss_o_2', 'Ast_poss_o_2', 'TO_poss_o_2', 'Stl_poss_o_2', 'Blk_poss_o_2', 'PF_poss_o_2', 'Score_poss_d_2', 'FGM_poss_d_2', 'FGA_poss_d_2', 'FGM3_poss_d_2', 'FGA3_poss_d_2', 'FTM_poss_d_2', 'FTA_poss_d_2', 'OR_poss_d_2', 'DR_poss_d_2', 'Ast_poss_d_2', 'TO_poss_d_2', 'Stl_poss_d_2', 'Blk_poss_d_2', 'PF_poss_d_2', 'sos_Score_poss_o_2', 'sos_FGM_poss_o_2', 'sos_FGA_poss_o_2', 'sos_FGM3_poss_o_2', 'sos_FGA3_poss_o_2', 'sos_FTM_poss_o_2', 'sos_FTA_poss_o_2', 'sos_OR_poss_o_2', 'sos_DR_poss_o_2', 'sos_Ast_poss_o_2', 'sos_TO_poss_o_2', 'sos_Stl_poss_o_2', 'sos_Blk_poss_o_2', 'sos_PF_poss_o_2', 'sos_Score_poss_d_2', 'sos_FGM_poss_d_2', 'sos_FGA_poss_d_2', 'sos_FGM3_poss_d_2', 'sos_FGA3_poss_d_2', 'sos_FTM_poss_d_2', 'sos_FTA_poss_d_2', 'sos_OR_poss_d_2', 'sos_DR_poss_d_2', 'sos_Ast_poss_d_2', 'sos_TO_poss_d_2', 'sos_Stl_poss_d_2', 'sos_Blk_poss_d_2', 'sos_PF_poss_d_2']



```python
X_df = train.drop(columns=["Season", "DayNum", "TeamID_1", "TeamID_2", "Margin"])
print(f"X_df: {str(X_df.shape):>24}")

X = torch.as_tensor(
    StandardScaler().fit_transform(X_df.values),
    dtype=torch.float32,
    device=device,
)

print(f"X:    {X.shape}")

y_s = train["Margin"]
print(f"y_s: {str(y_s.shape):>21}")
scaler_y = StandardScaler()

y = torch.tensor(
    scaler_y.fit_transform(train[["Margin"]]).flatten(),
    dtype=torch.float32,
    device=device,
)

print(f"y:    {y.shape}")
```

    X_df:            (202033, 112)
    X:    torch.Size([202033, 112])
    y_s:             (202033,)
    y:    torch.Size([202033])



```python
def brier_score(y_pred_np, y_true_s):
    pred_win_prob = 1 / (1 + np.exp(-y_pred_np * 0.1))
    team_1_won = (y_true_s.values > 0).astype(float)
    return np.mean((pred_win_prob - team_1_won) ** 2)
```


```python
params = {
    "tree_method": "hist",
    "device": "gpu",
    "max_depth": 3,
    "colsample_bytree": 0.5,
    "subsample": 0.8,
    "eta": 0.02,
    "min_child_weight": 80,
    "verbosity": 1,
}

print(f"xgboost")
y_pred_oof = np.zeros(y_s.shape[0])
y_pred_oof2 = np.zeros(y_s.shape[0])
    
for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X_df.index), 1):
    print(f"  fold {fold_n}")
    dm_fold = xgb.DMatrix(X_df.iloc[i_fold], label=y_s.iloc[i_fold])
    dm_oof = xgb.DMatrix(X_df.iloc[i_oof], label=y_s.iloc[i_oof])

    print("  xgb.train")
    m = xgb.train(
        params,
        dm_fold,
        num_boost_round=2000,
        evals=[(dm_fold, "fold"), (dm_oof, "oof")],
        verbose_eval=250,
    )

    y_pred_oof[i_oof] = m.predict(dm_oof)
    
    print("  XGBRegressor")
    m = xgb.XGBRegressor(
        tree_method="hist",
        device="cuda",
        max_depth=3,
        colsample_bytree=0.5,
        subsample=0.8,
        n_estimators=2000,
        learning_rate=0.02,
        min_child_weight=80,
        verbosity=1,
    )
    
    X_fold = cudf.DataFrame.from_pandas(X_df.iloc[i_fold])
    y_fold = cudf.Series(y_s.iloc[i_fold])
    X_oof = cudf.DataFrame.from_pandas(X_df.iloc[i_oof])
    y_oof = cudf.Series(y_s.iloc[i_oof])
    
    m.fit(
        X_fold,
        y_fold,
        verbose=250,
        eval_set=[
            (X_fold, y_fold),
            (X_oof, y_oof)
        ],
    )
    
    y_pred_oof2[i_oof] = m.predict(X_oof)
    
    print()

score = brier_score(y_pred_oof, y_s)
print(f"  score: {score:.4f}")
score = brier_score(y_pred_oof2, y_s)
print(f"  score: {score:.4f}")
```

    xgboost
      fold 1
      xgb.train
    [0]	fold-rmse:16.41412	oof-rmse:16.49274
    [250]	fold-rmse:11.81616	oof-rmse:11.91527
    [500]	fold-rmse:11.23284	oof-rmse:11.35894
    [750]	fold-rmse:11.05311	oof-rmse:11.20042
    [1000]	fold-rmse:10.97224	oof-rmse:11.13901
    [1250]	fold-rmse:10.92507	oof-rmse:11.10907
    [1500]	fold-rmse:10.89068	oof-rmse:11.09109
    [1750]	fold-rmse:10.86242	oof-rmse:11.07966
    [1999]	fold-rmse:10.83845	oof-rmse:11.07211
      XGBRegressor
    [0]	validation_0-rmse:16.41412	validation_1-rmse:16.49274
    [250]	validation_0-rmse:11.81616	validation_1-rmse:11.91527
    [500]	validation_0-rmse:11.23284	validation_1-rmse:11.35894
    [750]	validation_0-rmse:11.05311	validation_1-rmse:11.20042
    [1000]	validation_0-rmse:10.97224	validation_1-rmse:11.13901
    [1250]	validation_0-rmse:10.92507	validation_1-rmse:11.10907
    [1500]	validation_0-rmse:10.89068	validation_1-rmse:11.09109
    [1750]	validation_0-rmse:10.86242	validation_1-rmse:11.07966
    [1999]	validation_0-rmse:10.83845	validation_1-rmse:11.07211
    
      fold 2
      xgb.train
    [0]	fold-rmse:16.41678	oof-rmse:16.44453
    [250]	fold-rmse:11.82918	oof-rmse:11.83978
    [500]	fold-rmse:11.24746	oof-rmse:11.27648
    [750]	fold-rmse:11.06912	oof-rmse:11.11724
    [1000]	fold-rmse:10.99009	oof-rmse:11.05728
    [1250]	fold-rmse:10.94168	oof-rmse:11.02836
    [1500]	fold-rmse:10.90663	oof-rmse:11.01255
    [1750]	fold-rmse:10.87854	oof-rmse:11.00239
    [1999]	fold-rmse:10.85455	oof-rmse:10.99577
      XGBRegressor
    [0]	validation_0-rmse:16.41678	validation_1-rmse:16.44453
    [250]	validation_0-rmse:11.82918	validation_1-rmse:11.83978
    [500]	validation_0-rmse:11.24746	validation_1-rmse:11.27648
    [750]	validation_0-rmse:11.06912	validation_1-rmse:11.11724
    [1000]	validation_0-rmse:10.99009	validation_1-rmse:11.05728
    [1250]	validation_0-rmse:10.94168	validation_1-rmse:11.02836
    [1500]	validation_0-rmse:10.90663	validation_1-rmse:11.01255
    [1750]	validation_0-rmse:10.87854	validation_1-rmse:11.00239
    [1999]	validation_0-rmse:10.85455	validation_1-rmse:10.99577
    
      fold 3
      xgb.train
    [0]	fold-rmse:16.43755	oof-rmse:16.40973
    [250]	fold-rmse:11.81925	oof-rmse:11.91708
    [500]	fold-rmse:11.23273	oof-rmse:11.34793
    [750]	fold-rmse:11.05455	oof-rmse:11.18916
    [1000]	fold-rmse:10.97384	oof-rmse:11.12728
    [1250]	fold-rmse:10.92673	oof-rmse:11.09887
    [1500]	fold-rmse:10.89302	oof-rmse:11.08192
    [1750]	fold-rmse:10.86512	oof-rmse:11.07105
    [1999]	fold-rmse:10.84202	oof-rmse:11.06395
      XGBRegressor
    [0]	validation_0-rmse:16.43755	validation_1-rmse:16.40973
    [250]	validation_0-rmse:11.81925	validation_1-rmse:11.91708
    [500]	validation_0-rmse:11.23273	validation_1-rmse:11.34793
    [750]	validation_0-rmse:11.05455	validation_1-rmse:11.18916
    [1000]	validation_0-rmse:10.97384	validation_1-rmse:11.12728
    [1250]	validation_0-rmse:10.92673	validation_1-rmse:11.09887
    [1500]	validation_0-rmse:10.89302	validation_1-rmse:11.08192
    [1750]	validation_0-rmse:10.86512	validation_1-rmse:11.07105
    [1999]	validation_0-rmse:10.84202	validation_1-rmse:11.06395
    
      fold 4
      xgb.train
    [0]	fold-rmse:16.44885	oof-rmse:16.36198
    [250]	fold-rmse:11.82601	oof-rmse:11.84448
    [500]	fold-rmse:11.24018	oof-rmse:11.31309
    [750]	fold-rmse:11.06067	oof-rmse:11.16409
    [1000]	fold-rmse:10.98027	oof-rmse:11.10693
    [1250]	fold-rmse:10.93366	oof-rmse:11.07866
    [1500]	fold-rmse:10.89967	oof-rmse:11.06103
    [1750]	fold-rmse:10.87201	oof-rmse:11.04947
    [1999]	fold-rmse:10.84942	oof-rmse:11.04161
      XGBRegressor
    [0]	validation_0-rmse:16.44885	validation_1-rmse:16.36198
    [250]	validation_0-rmse:11.82601	validation_1-rmse:11.84448
    [500]	validation_0-rmse:11.24018	validation_1-rmse:11.31309
    [750]	validation_0-rmse:11.06067	validation_1-rmse:11.16409
    [1000]	validation_0-rmse:10.98027	validation_1-rmse:11.10693
    [1250]	validation_0-rmse:10.93366	validation_1-rmse:11.07866
    [1500]	validation_0-rmse:10.89967	validation_1-rmse:11.06103
    [1750]	validation_0-rmse:10.87201	validation_1-rmse:11.04947
    [1999]	validation_0-rmse:10.84942	validation_1-rmse:11.04161
    
      fold 5
      xgb.train
    [0]	fold-rmse:16.42735	oof-rmse:16.43855
    [250]	fold-rmse:11.83235	oof-rmse:11.88605
    [500]	fold-rmse:11.24305	oof-rmse:11.32131
    [750]	fold-rmse:11.06342	oof-rmse:11.16125
    [1000]	fold-rmse:10.98315	oof-rmse:11.09827
    [1250]	fold-rmse:10.93662	oof-rmse:11.06861
    [1500]	fold-rmse:10.90162	oof-rmse:11.05145
    [1750]	fold-rmse:10.87281	oof-rmse:11.04075
    [1999]	fold-rmse:10.84913	oof-rmse:11.03351
      XGBRegressor
    [0]	validation_0-rmse:16.42735	validation_1-rmse:16.43855
    [250]	validation_0-rmse:11.83235	validation_1-rmse:11.88605
    [500]	validation_0-rmse:11.24305	validation_1-rmse:11.32131
    [750]	validation_0-rmse:11.06342	validation_1-rmse:11.16125
    [1000]	validation_0-rmse:10.98315	validation_1-rmse:11.09827
    [1250]	validation_0-rmse:10.93662	validation_1-rmse:11.06861
    [1500]	validation_0-rmse:10.90162	validation_1-rmse:11.05145
    [1750]	validation_0-rmse:10.87281	validation_1-rmse:11.04075
    [1999]	validation_0-rmse:10.84913	validation_1-rmse:11.03351
    
      score: 0.1660
      score: 0.1660



```python
print("torch")
n_epochs = 1_000
hidden_size = 64
loss_fn = torch.nn.MSELoss()

y_pred_oof = torch.zeros(
    y.shape[0],
    dtype=torch.float32,
    requires_grad=False,
    device=device,
)

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X_df.index), 1):
    print(f"  fold {fold_n}")

    weights1 = torch.nn.Parameter(
        0.1 * torch.randn(X_df.shape[1], hidden_size, device=device)
    )
    bias1 = torch.nn.Parameter(torch.zeros(hidden_size, device=device))
    weights2 = torch.nn.Parameter(0.1 * torch.randn(hidden_size, 1, device=device))
    bias2 = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.Adam([weights1, bias1, weights2, bias2], weight_decay=1e-4)

    for epoch_n in range(1, n_epochs + 1):
        y_pred_fold_epoch = F.leaky_relu(X[i_fold] @ weights1 + bias1, negative_slope=0.1) @ weights2 + bias2
        loss_fold_epoch = loss_fn(y_pred_fold_epoch, y[i_fold].view(-1, 1))
        optimizer.zero_grad()
        loss_fold_epoch.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_oof_epoch = F.leaky_relu(X[i_oof] @ weights1 + bias1, negative_slope=0.1) @ weights2 + bias2
            loss_oof_epoch = loss_fn(y_pred_oof_epoch, y[i_oof].view(-1, 1))

        if epoch_n > (n_epochs - 3):
            print(
                f"    epoch {epoch_n:>6}: "
                f"fold={loss_fold_epoch.item():.4f} "
                f"oof={loss_oof_epoch.item():.4f}"
            )

    with torch.no_grad():
        y_pred_oof[i_oof] = (
            F.leaky_relu(X[i_oof] @ weights1 + bias1, negative_slope=0.1) @ weights2 + bias2
        ).flatten()

    print()

y_pred_oof = scaler_y.inverse_transform(
    y_pred_oof.cpu().numpy().reshape(-1, 1)
).flatten()

score = brier_score(y_pred_oof, y_s)
print(f"  score: {score:.4f}")
```

    torch
      fold 1
        epoch    998: fold=0.4326 oof=0.4466
        epoch    999: fold=0.4326 oof=0.4466
        epoch   1000: fold=0.4326 oof=0.4466
    
      fold 2
        epoch    998: fold=0.4357 oof=0.4407
        epoch    999: fold=0.4357 oof=0.4407
        epoch   1000: fold=0.4357 oof=0.4407
    
      fold 3
        epoch    998: fold=0.4333 oof=0.4437
        epoch    999: fold=0.4333 oof=0.4438
        epoch   1000: fold=0.4333 oof=0.4437
    
      fold 4
        epoch    998: fold=0.4329 oof=0.4422
        epoch    999: fold=0.4329 oof=0.4422
        epoch   1000: fold=0.4328 oof=0.4422
    
      fold 5
        epoch    998: fold=0.4328 oof=0.4433
        epoch    999: fold=0.4328 oof=0.4434
        epoch   1000: fold=0.4328 oof=0.4433
    
      score: 0.1649



```python

```
