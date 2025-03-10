#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    del input_dir, data_dir, SampleSubmissionStage2
except NameError:
    pass

import numpy as np
import pandas as pd

input_dir = "../datasets"
data_dir = f"{input_dir}/march-machine-learning-mania-2025"
SampleSubmissionStage2 = pd.read_csv(f"{data_dir}/SampleSubmissionStage2.csv")

SampleSubmissionStage2[["Season", "TeamID_1", "TeamID_2"]] = (
    SampleSubmissionStage2["ID"].str.split("_", expand=True).astype("int32")
)

SampleSubmissionStage2 = SampleSubmissionStage2.drop(columns="ID")
SampleSubmissionStage2 = SampleSubmissionStage2.drop(columns="Pred")
print(f"SampleSubmissionStage2: {SampleSubmissionStage2.shape}")


# In[2]:


try:
    del season_2025
except NameError:
    pass

print(f"reading season_2025.csv")
season_2025 = pd.read_csv(f"{input_dir}/net/season_2025.csv")

season_2025 = pd.concat(
    [
        season_2025.select_dtypes("int64").astype("int32"),
        season_2025.select_dtypes("float64").astype("float32"),
    ],
    axis=1,
)

print(f"train: {str(season_2025.shape):>23}")


# In[3]:


print(SampleSubmissionStage2.columns.to_list())
print(season_2025.columns.to_list()[:7])


# In[4]:


try:
    del sub
except NameError:
    pass

sub = pd.merge(
    SampleSubmissionStage2,
    season_2025,
    left_on=["Season", "TeamID_1"],
    right_on=["Season", "TeamID"],
)

sub = sub.drop(columns="TeamID")

sub = sub.rename(
    columns={c: f"{c}_1" for c in sub if c not in ("Season", "TeamID_1", "TeamID_2")}
)

sub = pd.merge(
    sub,
    season_2025,
    left_on=["Season", "TeamID_2"],
    right_on=["Season", "TeamID"],
)

sub = sub.drop(columns="TeamID")

sub = sub.rename(
    columns={
        c: f"{c}_2"
        for c in sub
        if c not in ("Season", "TeamID_1", "TeamID_2") and not c.endswith("_1")
    }
)

print(sub.columns.to_list()[:7])
print(sub.columns.to_list()[-5:])


# In[ ]:
