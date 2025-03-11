```python
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

kfold = KFold(shuffle=True, random_state=42)
```


```python
print(f"reading train.csv")
train = pd.read_csv("../datasets/net/train.csv")

train = pd.concat(
    [
        train.select_dtypes("int64").astype("int32"),
        train.select_dtypes("float64").astype("float32"),
    ],
    axis=1,
)

print(f"\ntrain: {str(train.shape):>23}")
```

    reading train.csv
    
    train:           (202033, 413)



```python
X_df = train.drop(columns=["Season", "DayNum", "TeamID_1", "TeamID_2", "Margin"])
print(f"X_df: {str(X_df.shape):>24}")

X = torch.as_tensor(
    StandardScaler().fit_transform(X_df.values),
    dtype=torch.float32,
    device=device,
)

print(f"X:    {X.shape}")

y_df = train[["Margin"]]
print(f"y_df: {str(y_df.shape):>22}")
scaler_y = StandardScaler()

y = torch.tensor(
    scaler_y.fit_transform(y_df).flatten(),
    dtype=torch.float32,
    device=device,
)

print(f"y:    {y.shape}")
```

    X_df:            (202033, 408)
    X:    torch.Size([202033, 408])
    y_df:            (202033, 1)
    y:    torch.Size([202033])



```python
def brier_score(y_pred_np, y_true_df):
    pred_win_prob = 1 / (1 + np.exp(-y_pred_np * 0.1))
    team_1_won = (y_true_df.squeeze().values > 0).astype(float)
    return np.mean((pred_win_prob - team_1_won) ** 2)
```


```python
print(f"xgboost")
y_pred_oof = np.zeros(y_df.shape[0])

if device == "cuda":
    params = {
        "tree_method": "gpu_hist",
        "gpu_id": 0,
    }
else:
    params = {
        "tree_method": "hist",
        "nthread": 1,
    }

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X_df.index), 1):
    print(f"  fold {fold_n}")
    dm_fold = xgb.DMatrix(X_df.iloc[i_fold], label=y_df.iloc[i_fold])
    dm_oof = xgb.DMatrix(X_df.iloc[i_oof], label=y_df.iloc[i_oof])

    m = xgb.train(
        params,
        dm_fold,
        evals=[(dm_fold, "fold"), (dm_oof, "oof")],
    )

    y_pred_oof[i_oof] = m.predict(dm_oof)
    print()

score = brier_score(y_pred_oof, y_df)
print(f"  score: {score:.4f}")
```

    xgboost
      fold 1
    [0]	fold-rmse:14.37691	oof-rmse:14.47013
    [1]	fold-rmse:13.06497	oof-rmse:13.18169
    [2]	fold-rmse:12.26574	oof-rmse:12.40069
    [3]	fold-rmse:11.78441	oof-rmse:11.94179
    [4]	fold-rmse:11.47450	oof-rmse:11.65231
    [5]	fold-rmse:11.27353	oof-rmse:11.46340
    [6]	fold-rmse:11.13629	oof-rmse:11.34290
    [7]	fold-rmse:11.04355	oof-rmse:11.26369
    [8]	fold-rmse:10.97402	oof-rmse:11.20991
    [9]	fold-rmse:10.92515	oof-rmse:11.18040
    
      fold 2
    [0]	fold-rmse:14.37950	oof-rmse:14.41955
    [1]	fold-rmse:13.06958	oof-rmse:13.12387
    [2]	fold-rmse:12.27561	oof-rmse:12.34734
    [3]	fold-rmse:11.79247	oof-rmse:11.87604
    [4]	fold-rmse:11.48247	oof-rmse:11.58133
    [5]	fold-rmse:11.28185	oof-rmse:11.38719
    [6]	fold-rmse:11.14640	oof-rmse:11.26985
    [7]	fold-rmse:11.05062	oof-rmse:11.18991
    [8]	fold-rmse:10.98461	oof-rmse:11.13828
    [9]	fold-rmse:10.93357	oof-rmse:11.10145
    
      fold 3
    [0]	fold-rmse:14.37980	oof-rmse:14.41704
    [1]	fold-rmse:13.08908	oof-rmse:13.16835
    [2]	fold-rmse:12.27070	oof-rmse:12.37035
    [3]	fold-rmse:11.77818	oof-rmse:11.89763
    [4]	fold-rmse:11.47501	oof-rmse:11.62079
    [5]	fold-rmse:11.28153	oof-rmse:11.44424
    [6]	fold-rmse:11.15003	oof-rmse:11.32647
    [7]	fold-rmse:11.04836	oof-rmse:11.23868
    [8]	fold-rmse:10.98136	oof-rmse:11.18543
    [9]	fold-rmse:10.93431	oof-rmse:11.14999
    
      fold 4
    [0]	fold-rmse:14.39180	oof-rmse:14.32560
    [1]	fold-rmse:13.08738	oof-rmse:13.07616
    [2]	fold-rmse:12.28062	oof-rmse:12.30161
    [3]	fold-rmse:11.79244	oof-rmse:11.84008
    [4]	fold-rmse:11.48058	oof-rmse:11.55537
    [5]	fold-rmse:11.28227	oof-rmse:11.38059
    [6]	fold-rmse:11.14482	oof-rmse:11.26346
    [7]	fold-rmse:11.04916	oof-rmse:11.18759
    [8]	fold-rmse:10.98364	oof-rmse:11.14192
    [9]	fold-rmse:10.93436	oof-rmse:11.10624
    
      fold 5
    [0]	fold-rmse:14.40060	oof-rmse:14.43940
    [1]	fold-rmse:13.09054	oof-rmse:13.15637
    [2]	fold-rmse:12.27573	oof-rmse:12.36175
    [3]	fold-rmse:11.78286	oof-rmse:11.89750
    [4]	fold-rmse:11.47332	oof-rmse:11.60062
    [5]	fold-rmse:11.28073	oof-rmse:11.41922
    [6]	fold-rmse:11.13948	oof-rmse:11.29112
    [7]	fold-rmse:11.04103	oof-rmse:11.20773
    [8]	fold-rmse:10.97372	oof-rmse:11.15453
    [9]	fold-rmse:10.92747	oof-rmse:11.12282
    
      score: 0.1686



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
    bias2 = torch.nn.Parameter(torch.zeros(y_df.shape[1], device=device))
    optimizer = torch.optim.Adam([weights1, bias1, weights2, bias2], weight_decay=1e-4)

    for epoch_n in range(1, n_epochs + 1):
        y_pred_fold_epoch = F.relu(X[i_fold] @ weights1 + bias1) @ weights2 + bias2
        loss_fold_epoch = loss_fn(y_pred_fold_epoch, y[i_fold].view(-1, 1))
        optimizer.zero_grad()
        loss_fold_epoch.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_oof_epoch = F.relu(X[i_oof] @ weights1 + bias1) @ weights2 + bias2
            loss_oof_epoch = loss_fn(y_pred_oof_epoch, y[i_oof].view(-1, 1))

        if epoch_n > (n_epochs - 3):
            print(
                f"    epoch {epoch_n:>6}: "
                f"fold={loss_fold_epoch.item():.4f} "
                f"oof={loss_oof_epoch.item():.4f}"
            )

    with torch.no_grad():
        y_pred_oof[i_oof] = (
            F.relu(X[i_oof] @ weights1 + bias1) @ weights2 + bias2
        ).flatten()

    print()

y_pred_oof = scaler_y.inverse_transform(
    y_pred_oof.cpu().numpy().reshape(-1, 1)
).flatten()

score = brier_score(y_pred_oof, y_df)
print(f"  score: {score:.4f}")
```

    torch
      fold 1
        epoch    998: fold=0.4254 oof=0.4504
        epoch    999: fold=0.4254 oof=0.4504
        epoch   1000: fold=0.4254 oof=0.4504
    
      fold 2
        epoch    998: fold=0.4280 oof=0.4437
        epoch    999: fold=0.4280 oof=0.4437
        epoch   1000: fold=0.4280 oof=0.4437
    
      fold 3
        epoch    998: fold=0.4257 oof=0.4465
        epoch    999: fold=0.4257 oof=0.4465
        epoch   1000: fold=0.4257 oof=0.4465
    
      fold 4
        epoch    998: fold=0.4264 oof=0.4455
        epoch    999: fold=0.4263 oof=0.4455
        epoch   1000: fold=0.4263 oof=0.4455
    
      fold 5
        epoch    998: fold=0.4253 oof=0.4453
        epoch    999: fold=0.4252 oof=0.4454
        epoch   1000: fold=0.4252 oof=0.4454
    
      score: 0.1658



```python

```
