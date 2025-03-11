#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


def brier_score(y_pred_np, y_true_df):
    pred_win_prob = 1 / (1 + np.exp(-y_pred_np * 0.25))
    team_1_won = (y_true_df.squeeze().values > 0).astype(float)
    return np.mean((pred_win_prob - team_1_won) ** 2)


# In[ ]:


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


# In[ ]:


print("torch")
n_epochs = 100
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
    weights1 = 0.1 * torch.randn(X_df.shape[1], hidden_size, device=device)
    bias1 = torch.zeros(hidden_size, requires_grad=True, device=device)
    weights2 = 0.1 * torch.randn(hidden_size, 1, device=device)
    bias2 = torch.zeros(y_df.shape[1], requires_grad=True, device=device)
    optimizer = torch.optim.Adam([weights1, bias1, weights2, bias2])

    for epoch_n in range(1, n_epochs + 1):
        y_pred_fold_epoch = F.relu(X[i_fold] @ weights1 + bias1) @ weights2 + bias2
        loss_fold_epoch = loss_fn(y_pred_fold_epoch, y[i_fold].view(-1, 1))
        optimizer.zero_grad()
        loss_fold_epoch.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_oof_epoch = F.relu(X[i_oof] @ weights1 + bias1) @ weights2 + bias2
            loss_oof_epoch = loss_fn(y_pred_oof_epoch, y[i_oof].view(-1, 1))

        if epoch_n % (n_epochs // 10) == 0:
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
print(f"  score: {score.item():.4f}")


# In[ ]:
