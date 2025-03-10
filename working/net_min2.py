#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd

try:
    del train
except NameError:
    pass

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


# In[33]:


import torch

try:
    del device
except NameError:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"


# In[34]:


from sklearn.preprocessing import StandardScaler

try:
    del X_df, X, y_df, y
except NameError:
    pass

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


# In[35]:


import numpy as np


def brier_score(y_pred, y_true):
    win_prob = 1 / (1 + np.exp(-y_pred * 0.25))
    team_1_won = (y_true > 0).astype(float)
    return np.mean((win_prob - team_1_won) ** 2)


# In[36]:


from sklearn.model_selection import KFold

try:
    del kfold
except NameError:
    pass

kfold = KFold(shuffle=True, random_state=42)


# In[37]:


import torch.nn.functional as F

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
    print(f"\nfold {fold_n}")

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
                f"  epoch {epoch_n:>6}: "
                f"fold={loss_fold_epoch.item():.4f} "
                f"oof={loss_oof_epoch.item():.4f}"
            )

    with torch.no_grad():
        y_pred_oof[i_oof] = (
            F.relu(X[i_oof] @ weights1 + bias1) @ weights2 + bias2
        ).flatten()

y_pred_oof = scaler_y.inverse_transform(
    y_pred_oof.cpu().numpy().reshape(-1, 1)
).flatten()
score = brier_score(y_pred_oof, y_df.squeeze())
print(f"\nScore: {score.item():.4f}")


# In[ ]:
