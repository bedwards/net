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

    Collecting gputil
      Downloading GPUtil-1.4.0.tar.gz (5.5 kB)
      Preparing metadata (setup.py) ... [?25l[?25hdone
    Building wheels for collected packages: gputil
      Building wheel for gputil (setup.py) ... [?25l[?25hdone
      Created wheel for gputil: filename=GPUtil-1.4.0-py3-none-any.whl size=7392 sha256=c12e89fa2bcbd43d0a0518905ecf2b0e1b061b55d5a1a4431e69b1e086af0f0c
      Stored in directory: /root/.cache/pip/wheels/a9/8a/bd/81082387151853ab8b6b3ef33426e98f5cbfebc3c397a9d4d0
    Successfully built gputil
    Installing collected packages: gputil
    Successfully installed gputil-1.4.0



```python
import warnings

warnings.simplefilter("ignore")

import json
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
xgb_models = []
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

    xgb_models.append(m)
    y_pred_oof[i_oof] = m.predict(X_oof)
    GPUtil.showUtilization()
    print()

score = brier_score(y_pred_oof)
print(f"xgboost score: {score:.4f}")

for fold_n, m in enumerate(xgb_models, 1):
    fn = f"xgb_{f'{score:.4f}'[2:6]}_{fold_n}.json"
    m.get_booster().save_model(fn)
    print(f"wrote {fn}")
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
    |  0 |  0% |  1% |
    |  1 |  0% |  0% |
    
      fold 2
    [0]	validation_0-rmse:16.41678	validation_1-rmse:16.44453
    [500]	validation_0-rmse:11.24746	validation_1-rmse:11.27648
    [1000]	validation_0-rmse:10.99009	validation_1-rmse:11.05728
    [1500]	validation_0-rmse:10.90663	validation_1-rmse:11.01255
    [1999]	validation_0-rmse:10.85455	validation_1-rmse:10.99577
    | ID | GPU | MEM |
    ------------------
    |  0 | 23% |  2% |
    |  1 |  0% |  0% |
    
      fold 3
    [0]	validation_0-rmse:16.43755	validation_1-rmse:16.40973
    [500]	validation_0-rmse:11.23273	validation_1-rmse:11.34793
    [1000]	validation_0-rmse:10.97384	validation_1-rmse:11.12728
    [1500]	validation_0-rmse:10.89302	validation_1-rmse:11.08192
    [1999]	validation_0-rmse:10.84202	validation_1-rmse:11.06395
    | ID | GPU | MEM |
    ------------------
    |  0 | 36% |  2% |
    |  1 |  0% |  0% |
    
      fold 4
    [0]	validation_0-rmse:16.44885	validation_1-rmse:16.36198
    [500]	validation_0-rmse:11.24018	validation_1-rmse:11.31309
    [1000]	validation_0-rmse:10.98027	validation_1-rmse:11.10693
    [1500]	validation_0-rmse:10.89967	validation_1-rmse:11.06103
    [1999]	validation_0-rmse:10.84942	validation_1-rmse:11.04161
    | ID | GPU | MEM |
    ------------------
    |  0 | 43% |  2% |
    |  1 |  0% |  0% |
    
      fold 5
    [0]	validation_0-rmse:16.42735	validation_1-rmse:16.43855
    [500]	validation_0-rmse:11.24305	validation_1-rmse:11.32131
    [1000]	validation_0-rmse:10.98315	validation_1-rmse:11.09827
    [1500]	validation_0-rmse:10.90162	validation_1-rmse:11.05145
    [1999]	validation_0-rmse:10.84913	validation_1-rmse:11.03351
    | ID | GPU | MEM |
    ------------------
    |  0 | 49% |  2% |
    |  1 |  0% |  0% |
    
    xgboost score: 0.1660
    wrote xgb_1660_1.json
    wrote xgb_1660_2.json
    wrote xgb_1660_3.json
    wrote xgb_1660_4.json
    wrote xgb_1660_5.json



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


def weight(*size):
    return torch.nn.Parameter(0.1 * torch.randn(*size, device=device))


def bias(*size):
    return torch.nn.Parameter(torch.zeros(*size, device=device))


mse_ = torch.nn.MSELoss()


def mse(y_pred_epoch, i):
    return mse_(y_pred_epoch, y[i].view(-1, 1))


def aslist(param):
    return param.cpu().detach().numpy().tolist()


def aspy(m):
    return {
        "w": [aslist(w) for w in m["w"]],
        "b": [aslist(b) for b in m["b"]],
    }


n_epochs = 1_000
hidden_size = 64
torch_models = []

y_pred_oof = torch.zeros(
    y.shape[0],
    dtype=torch.float32,
    requires_grad=False,
    device=device,
)

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X_df.index), 1):
    print(f"  fold {fold_n}")

    m = {
        "w": [
            weight(X_df.shape[1], hidden_size),
            weight(hidden_size, 1),
        ],
        "b": [
            bias(hidden_size),
            bias(1),
        ],
    }

    optimizer = torch.optim.Adam(m["w"] + m["b"], weight_decay=1e-4)

    def forward(i):
        return (
            F.leaky_relu(X[i] @ m["w"][0] + m["b"][0], negative_slope=0.1) @ m["w"][1]
            + m["b"][1]
        )

    for epoch_n in range(1, n_epochs + 1):
        y_pred_epoch_fold = forward(i_fold)
        mse_epoch_fold = mse(y_pred_epoch_fold, i_fold)
        optimizer.zero_grad()
        mse_epoch_fold.backward()
        optimizer.step()

        if (epoch_n % (n_epochs // 2) == 0) or (epoch_n > (n_epochs - 3)):
            with torch.no_grad():
                y_pred_epoch_oof = forward(i_oof)
                mse_epoch_oof = mse(y_pred_epoch_oof, i_oof)

            print(
                f"    epoch {epoch_n:>6}: "
                f"fold={mse_epoch_fold.item():.4f} "
                f"oof={mse_epoch_oof.item():.4f}"
            )

    with torch.no_grad():
        y_pred_oof[i_oof] = forward(i_oof).flatten()

    GPUtil.showUtilization()
    torch_models.append(aspy(m))
    print()

y_pred_oof = scaler_y.inverse_transform(
    y_pred_oof.cpu().numpy().reshape(-1, 1)
).flatten()

score = brier_score(y_pred_oof)
print(f"torch score:   {score:.4f}")

for fold_n, m in enumerate(torch_models, 1):
    fn = f"nn_{f'{score:.4f}'[2:6]}_{fold_n}.json"
    with open(fn, "w") as f:
        json.dump(m, f)
    print(f"wrote {fn}")
```

    torch
    X:    torch.Size([202033, 112])
    y:    torch.Size([202033])
      fold 1
        epoch    500: fold=0.4375 oof=0.4466
        epoch    998: fold=0.4324 oof=0.4458
        epoch    999: fold=0.4324 oof=0.4457
        epoch   1000: fold=0.4324 oof=0.4459
    | ID | GPU | MEM |
    ------------------
    |  0 | 91% |  4% |
    |  1 |  0% |  0% |
    
      fold 2
        epoch    500: fold=0.4393 oof=0.4413
        epoch    998: fold=0.4341 oof=0.4404
        epoch    999: fold=0.4341 oof=0.4405
        epoch   1000: fold=0.4341 oof=0.4404
    | ID | GPU | MEM |
    ------------------
    |  0 | 91% |  4% |
    |  1 |  0% |  0% |
    
      fold 3
        epoch    500: fold=0.4405 oof=0.4473
        epoch    998: fold=0.4345 oof=0.4445
        epoch    999: fold=0.4345 oof=0.4445
        epoch   1000: fold=0.4345 oof=0.4445
    | ID | GPU | MEM |
    ------------------
    |  0 | 90% |  4% |
    |  1 |  0% |  0% |
    
      fold 4
        epoch    500: fold=0.4390 oof=0.4445
        epoch    998: fold=0.4337 oof=0.4431
        epoch    999: fold=0.4336 oof=0.4432
        epoch   1000: fold=0.4336 oof=0.4431
    | ID | GPU | MEM |
    ------------------
    |  0 | 90% |  4% |
    |  1 |  0% |  0% |
    
      fold 5
        epoch    500: fold=0.4394 oof=0.4433
        epoch    998: fold=0.4341 oof=0.4420
        epoch    999: fold=0.4341 oof=0.4420
        epoch   1000: fold=0.4341 oof=0.4420
    | ID | GPU | MEM |
    ------------------
    |  0 | 91% |  4% |
    |  1 |  0% |  0% |
    
    torch score:   0.1648
    wrote nn_1648_1.json
    wrote nn_1648_2.json
    wrote nn_1648_3.json
    wrote nn_1648_4.json
    wrote nn_1648_5.json



```python

```
