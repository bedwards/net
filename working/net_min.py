#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd

train = pd.read_csv("../datasets/net/train.csv")
train = pd.concat(
    [
        train.select_dtypes("int64").astype("int32"),
        train.select_dtypes("float64").astype("float32"),
    ],
    axis=1,
)


# In[39]:


def margin_to_prob(margin):
    return 1 / (1 + torch.exp(-margin * 0.25))


def brier_score(probs, outcomes, chunk_size=10000):
    total_score = 0
    n_samples = probs.shape[0]

    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        chunk_score = torch.mean((probs[i:end] - outcomes[i:end]) ** 2)
        total_score += chunk_score * (end - i)

    return total_score / n_samples


# In[40]:


from sklearn.preprocessing import StandardScaler
import torch

X = train.drop(columns=["Season", "DayNum", "TeamID_1", "TeamID_2", "Margin"])
X = X.values
X = StandardScaler().fit_transform(X)
X = torch.as_tensor(X, dtype=torch.float32, device="cuda")

y = train["Margin"].values.reshape(-1, 1)
y = StandardScaler().fit_transform(y)
y = torch.as_tensor(y, dtype=torch.float32, device="cuda")


# In[41]:


from sklearn.model_selection import KFold
import torch.nn.functional as F

hidden_size = 64
loss_fn = torch.nn.MSELoss()
n_epochs = 1_000
n_folds = 5
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
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

score = brier_score(margin_to_prob(y_pred_oof), (y > 0).float())
print(f"Score: {score.item():.4f}")


# In[ ]:


# In[ ]:
