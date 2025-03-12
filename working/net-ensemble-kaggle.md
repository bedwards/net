```python
import os

IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
print(f"running on kaggle: {IS_KAGGLE}")
```

    running on kaggle: True



```python
if not IS_KAGGLE:
    !pip install --upgrade numpy pandas xgboost scikit-learn
    !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    !pip install \
        --extra-index-url=https://pypi.nvidia.com \
        "cudf-cu12==25.2.*" "dask-cudf-cu12==25.2.*" "cuml-cu12==25.2.*" \
        "cugraph-cu12==25.2.*" "nx-cugraph-cu12==25.2.*" "cuspatial-cu12==25.2.*" \
        "cuproj-cu12==25.2.*" "cuxfilter-cu12==25.2.*" "cucim-cu12==25.2.*" \
        "pylibraft-cu12==25.2.*" "raft-dask-cu12==25.2.*" "cuvs-cu12==25.2.*" \
        "nx-cugraph-cu12==25.2.*"

!pip install gputil
```

    Requirement already satisfied: gputil in /usr/local/lib/python3.10/dist-packages (1.4.0)



```python
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import cudf
import GPUtil

if torch.cuda.is_available():
    device = "cuda"
else:
    raise

datasets_dir = "../input" if IS_KAGGLE else "../datasets"
kfold = KFold(shuffle=True, random_state=42)
```


```python
fn = "train_poss.csv"
print(f"reading {fn}")
train = pd.read_csv(f"{datasets_dir}/net-dataset/{fn}")

train = pd.concat(
    [
        train.select_dtypes("int64").astype("int32"),
        train.select_dtypes("float64").astype("float32"),
    ],
    axis=1,
)
```

    reading train_poss.csv



```python
print("train")
int_cols = train.select_dtypes("int32").columns.to_list()
print(f"{len(int_cols):>3} {int_cols}")
float_cols = train.select_dtypes("float32").columns.to_list()
o_cols = [c for c in float_cols if c.split("_")[2] == "o"]
print(f"{len(o_cols):>3} {o_cols[:3]} ... {o_cols[-3:]}")
d_cols = [c for c in float_cols if c.split("_")[2] == "d"]
print(f"{len(d_cols):>3} {d_cols[:3]} ... {d_cols[-3:]}")
sos_o_cols = [c for c in float_cols if c.split("_")[3] == "o"]
print(f"{len(sos_o_cols):>3} {sos_o_cols[:3]} ... {sos_o_cols[-3:]}")
sos_d_cols = [c for c in float_cols if c.split("_")[3] == "d"]
print(f"{len(sos_d_cols):>3} {sos_d_cols[:3]} ... {sos_d_cols[-3:]}")
print("---")
print(
    f"{train.shape[1]} {len(int_cols) + len(o_cols) + len(d_cols) + len(sos_o_cols) + len(sos_d_cols)}"
)
```

    train
      5 ['Season', 'DayNum', 'TeamID_1', 'TeamID_2', 'Margin']
     28 ['Score_poss_o_1', 'FGM_poss_o_1', 'FGA_poss_o_1'] ... ['Stl_poss_o_2', 'Blk_poss_o_2', 'PF_poss_o_2']
     28 ['Score_poss_d_1', 'FGM_poss_d_1', 'FGA_poss_d_1'] ... ['Stl_poss_d_2', 'Blk_poss_d_2', 'PF_poss_d_2']
     28 ['sos_Score_poss_o_1', 'sos_FGM_poss_o_1', 'sos_FGA_poss_o_1'] ... ['sos_Stl_poss_o_2', 'sos_Blk_poss_o_2', 'sos_PF_poss_o_2']
     28 ['sos_Score_poss_d_1', 'sos_FGM_poss_d_1', 'sos_FGA_poss_d_1'] ... ['sos_Stl_poss_d_2', 'sos_Blk_poss_d_2', 'sos_PF_poss_d_2']
    ---
    117 117



```python
print(f"train: {str(train.shape):>23}")

X_df = train.drop(columns=["Season", "DayNum", "TeamID_1", "TeamID_2", "Margin"])
print(f"X_df: {str(X_df.shape):>24}")

y_s = train["Margin"]
print(f"y_s: {str(y_s.shape):>21}")
scaler_y = StandardScaler()
```

    train:           (202033, 117)
    X_df:            (202033, 112)
    y_s:             (202033,)



```python
def brier_score(y_pred_oof):
    win_prob_pred_oof = 1 / (1 + np.exp(-y_pred_oof * 0.1))
    team_1_won = (y_s > 0).astype("int32")
    return np.mean((win_prob_pred_oof - team_1_won) ** 2)
```


```python
print(f"xgboost")
y_pred_oof = np.zeros(y_s.shape[0])

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X_df.index), 1):
    print(f"  fold {fold_n}")
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
        verbose=500,
        eval_set=[(X_fold, y_fold), (X_oof, y_oof)],
    )

    y_pred_oof[i_oof] = m.predict(X_oof)
    GPUtil.showUtilization()
    print()

score = brier_score(y_pred_oof)
print(f"xgboost score: {score:.4f}")
```

    xgboost
      fold 1
    [0]	validation_0-rmse:16.41412	validation_1-rmse:16.49274
    [500]	validation_0-rmse:11.23284	validation_1-rmse:11.35894
    [1000]	validation_0-rmse:10.97224	validation_1-rmse:11.13901
    [1500]	validation_0-rmse:10.89068	validation_1-rmse:11.09109
    [1999]	validation_0-rmse:10.83845	validation_1-rmse:11.07211
    | ID | GPU | MEM |
    ------------------
    |  0 | 11% |  4% |
    |  1 |  0% |  0% |
    
      fold 2
    [0]	validation_0-rmse:16.41678	validation_1-rmse:16.44453
    [500]	validation_0-rmse:11.24746	validation_1-rmse:11.27648
    [1000]	validation_0-rmse:10.99009	validation_1-rmse:11.05728
    [1500]	validation_0-rmse:10.90663	validation_1-rmse:11.01255
    [1999]	validation_0-rmse:10.85455	validation_1-rmse:10.99577
    | ID | GPU | MEM |
    ------------------
    |  0 | 24% |  4% |
    |  1 |  0% |  0% |
    
      fold 3
    [0]	validation_0-rmse:16.43755	validation_1-rmse:16.40973
    [500]	validation_0-rmse:11.23273	validation_1-rmse:11.34793
    [1000]	validation_0-rmse:10.97384	validation_1-rmse:11.12728
    [1500]	validation_0-rmse:10.89302	validation_1-rmse:11.08192
    [1999]	validation_0-rmse:10.84202	validation_1-rmse:11.06395
    | ID | GPU | MEM |
    ------------------
    |  0 | 20% |  4% |
    |  1 |  0% |  0% |
    
      fold 4
    [0]	validation_0-rmse:16.44885	validation_1-rmse:16.36198
    [500]	validation_0-rmse:11.24018	validation_1-rmse:11.31309
    [1000]	validation_0-rmse:10.98027	validation_1-rmse:11.10693
    [1500]	validation_0-rmse:10.89967	validation_1-rmse:11.06103
    [1999]	validation_0-rmse:10.84942	validation_1-rmse:11.04161
    | ID | GPU | MEM |
    ------------------
    |  0 | 23% |  4% |
    |  1 |  0% |  0% |
    
      fold 5
    [0]	validation_0-rmse:16.42735	validation_1-rmse:16.43855
    [500]	validation_0-rmse:11.24305	validation_1-rmse:11.32131
    [1000]	validation_0-rmse:10.98315	validation_1-rmse:11.09827
    [1500]	validation_0-rmse:10.90162	validation_1-rmse:11.05145
    [1999]	validation_0-rmse:10.84913	validation_1-rmse:11.03351
    | ID | GPU | MEM |
    ------------------
    |  0 | 50% |  4% |
    |  1 |  0% |  0% |
    
    xgboost score: 0.1660



```python
print("torch")

X = torch.as_tensor(
    StandardScaler().fit_transform(X_df.values),
    dtype=torch.float32,
    device=device,
)
print(f"X:    {X.shape}")

y = torch.tensor(
    scaler_y.fit_transform(train[["Margin"]]).flatten(),
    dtype=torch.float32,
    device=device,
)
print(f"y:    {y.shape}")

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
        y_pred_fold_epoch = (
            F.leaky_relu(X[i_fold] @ weights1 + bias1, negative_slope=0.1) @ weights2
            + bias2
        )
        loss_fold_epoch = loss_fn(y_pred_fold_epoch, y[i_fold].view(-1, 1))
        optimizer.zero_grad()
        loss_fold_epoch.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_oof_epoch = (
                F.leaky_relu(X[i_oof] @ weights1 + bias1, negative_slope=0.1) @ weights2
                + bias2
            )
            loss_oof_epoch = loss_fn(y_pred_oof_epoch, y[i_oof].view(-1, 1))

        if epoch_n > (n_epochs - 3):
            print(
                f"    epoch {epoch_n:>6}: "
                f"fold={loss_fold_epoch.item():.4f} "
                f"oof={loss_oof_epoch.item():.4f}"
            )

    with torch.no_grad():
        y_pred_oof[i_oof] = (
            F.leaky_relu(X[i_oof] @ weights1 + bias1, negative_slope=0.1) @ weights2
            + bias2
        ).flatten()

    GPUtil.showUtilization()
    print()

y_pred_oof = scaler_y.inverse_transform(
    y_pred_oof.cpu().numpy().reshape(-1, 1)
).flatten()

score = brier_score(y_pred_oof)
print(f"torch score:   {score:.4f}")
```

    torch
    X:    torch.Size([202033, 112])
    y:    torch.Size([202033])
      fold 1
        epoch    998: fold=0.4324 oof=0.4462
        epoch    999: fold=0.4324 oof=0.4462
        epoch   1000: fold=0.4324 oof=0.4462
    | ID | GPU | MEM |
    ------------------
    |  0 | 87% |  4% |
    |  1 |  0% |  0% |
    
      fold 2
        epoch    998: fold=0.4340 oof=0.4400
        epoch    999: fold=0.4340 oof=0.4400
        epoch   1000: fold=0.4340 oof=0.4400
    | ID | GPU | MEM |
    ------------------
    |  0 | 88% |  4% |
    |  1 |  0% |  0% |
    
      fold 3
        epoch    998: fold=0.4330 oof=0.4436
        epoch    999: fold=0.4330 oof=0.4436
        epoch   1000: fold=0.4330 oof=0.4436
    | ID | GPU | MEM |
    ------------------
    |  0 | 88% |  4% |
    |  1 |  0% |  0% |
    
      fold 4
        epoch    998: fold=0.4325 oof=0.4433
        epoch    999: fold=0.4325 oof=0.4433
        epoch   1000: fold=0.4325 oof=0.4433
    | ID | GPU | MEM |
    ------------------
    |  0 | 89% |  4% |
    |  1 |  0% |  0% |
    
      fold 5
        epoch    998: fold=0.4315 oof=0.4420
        epoch    999: fold=0.4315 oof=0.4419
        epoch   1000: fold=0.4315 oof=0.4419
    | ID | GPU | MEM |
    ------------------
    |  0 | 89% |  4% |
    |  1 |  0% |  0% |
    
    torch score:   0.1648



```python

```
