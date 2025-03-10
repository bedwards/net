#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().run_line_magic("reset", "-f")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import xgboost as xgb
import torch
import torch.nn.functional as F


# In[22]:


train = pd.read_csv("../datasets/net/train.csv")
train = pd.concat(
    [
        train.select_dtypes("int64").astype("int32"),
        train.select_dtypes("float64").astype("float32"),
    ],
    axis=1,
)


# In[23]:


def margin_to_prob(margin):
    return 1 / (1 + np.exp(-margin * 0.25))


def brier_score(y_pred, y_true):
    probs = margin_to_prob(y_pred)
    outcomes = (y_true > 0).astype(float)
    return np.mean((probs - outcomes) ** 2)


# In[24]:


X_ = train.drop(columns=["Season", "DayNum", "TeamID_1", "TeamID_2", "Margin"])
X_ = X_.values
X_ = StandardScaler().fit_transform(X_)

y_orig = train["Margin"].values
y_scaler = StandardScaler()
y_ = y_scaler.fit_transform(y_orig.reshape(-1, 1)).flatten()


# In[25]:


n_folds = 5
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)


# In[26]:


y_pred_oof = np.zeros(y_.shape[0])

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X_)):
    print(f"XGBoost Fold {fold_n}")

    dtrain = xgb.DMatrix(X_[i_fold], label=y_[i_fold])
    dval = xgb.DMatrix(X_[i_oof], label=y_[i_oof])

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "gpu_hist",
        "gpu_id": 0,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=20,
    )

    y_pred_oof[i_oof] = model.predict(dval)

y_pred_oof = y_scaler.inverse_transform(y_pred_oof.reshape(-1, 1)).flatten()
score = brier_score(y_pred_oof, y_orig)
print(f"XGBoost score: {score:.4f}")


# In[27]:


X = torch.as_tensor(X_, dtype=torch.float32, device="cuda")
y = torch.as_tensor(y_, dtype=torch.float32, device="cuda")


# In[ ]:


hidden_size = 64
loss_fn = torch.nn.MSELoss()
n_epochs = 10_000
y_pred_oof = torch.zeros(y.shape[0], requires_grad=False, device="cuda")

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X)):
    print(f"Fold {fold_n}")

    weights1 = torch.randn(X.shape[1], hidden_size, device="cuda") * 0.1
    bias1 = torch.zeros(hidden_size, requires_grad=True, device="cuda")
    weights2 = torch.randn(hidden_size, 1, device="cuda") * 0.1
    bias2 = torch.zeros(1, requires_grad=True, device="cuda")
    optimizer = torch.optim.Adam([weights1, bias1, weights2, bias2], lr=0.001)

    for epoch in range(n_epochs):
        y_pred = F.relu(X[i_fold] @ weights1 + bias1) @ weights2 + bias2
        loss = loss_fn(y_pred, y[i_fold].view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_pred = F.relu(X[i_oof] @ weights1 + bias1) @ weights2 + bias2
            val_loss = loss_fn(val_pred, y[i_oof].view(-1, 1))

        if epoch % (n_epochs // 10) == 0:
            print(
                f"  Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
            )

    with torch.no_grad():
        y_pred_oof[i_oof] = (
            F.relu(X[i_oof] @ weights1 + bias1) @ weights2 + bias2
        ).flatten()

y_pred_oof = y_scaler.inverse_transform(
    y_pred_oof.cpu().numpy().reshape(-1, 1)
).flatten()
score = brier_score(y_pred_oof, y_orig)
print(f"Score: {score.item():.4f}")


# In[ ]:


# In[ ]:
