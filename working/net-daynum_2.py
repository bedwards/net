#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.simplefilter("ignore")

import os
import numpy as np
import pandas as pd

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 6)
pd.set_option("display.width", None)

data_dir = f"../datasets/march-machine-learning-mania-2025"


# In[ ]:


raw = pd.DataFrame()

for fn in ["MNCAATourney", "MRegularSeason", "WNCAATourney", "WRegularSeason"]:
    raw = pd.concat(
        [
            raw,
            pd.read_csv(
                f"{data_dir}/{fn}DetailedResults.csv",
            ),
        ]
    ).reset_index(drop=True)

for c in raw.select_dtypes("int"):
    raw[c] = raw[c].astype("int32")

print(f"raw {raw.shape}")


# In[ ]:


tar = raw[["Season", "DayNum", "WTeamID", "LTeamID"]]
tar["WMargin"] = raw["WScore"] - raw["LScore"]
tar["LMargin"] = -tar["WMargin"]

tar = pd.concat(
    [
        tar.rename(
            columns={
                "WTeamID": "TeamID",
                "LTeamID": "OppID",
                "WMargin": "Margin",
            }
        ).drop(columns="LMargin"),
        tar.rename(
            columns={
                "LTeamID": "TeamID",
                "WTeamID": "OppID",
                "LMargin": "Margin",
            }
        ).drop(columns="WMargin"),
    ]
).reset_index(drop=True)

print(f"tar {tar.shape}\n")
print(tar)


# In[ ]:


gam = raw.copy()
gam.loc[gam["WLoc"] == "A", "WLoc_"] = -1
gam.loc[gam["WLoc"] == "N", "WLoc_"] = 0
gam.loc[gam["WLoc"] == "H", "WLoc_"] = 1
gam = gam.drop(columns="WLoc").rename(columns={"WLoc_": "WLoc"})
gam["WLoc"] = gam["WLoc"].astype("int32")
gam["LLoc"] = -gam["WLoc"]

gam = pd.concat(
    [
        gam.rename(columns={c: f"{c[1:]}_gu" for c in gam if c[0] == "W"}).rename(
            columns={c: f"{c[1:]}_gt" for c in gam if c[0] == "L"}
        ),
        gam.rename(columns={c: f"{c[1:]}_gu" for c in gam if c[0] == "L"}).rename(
            columns={c: f"{c[1:]}_gt" for c in gam if c[0] == "W"}
        ),
    ]
).reset_index(drop=True)

gam = gam.rename(columns={"TeamID_gu": "TeamID", "TeamID_gt": "OppID"})
gam["NumOT_gu"] = gam["NumOT"]
gam = gam.drop(columns=["NumOT", "Loc_gt"])

print(f"gam {gam.shape}\n")
print(gam[[c for c in gam if c[-3:-1] != "_g"]])
print()
print(gam[[c for c in gam if c[-3:] == "_gu"]])
print()
print(gam[[c for c in gam if c[-3:] == "_gt"]])


# In[ ]:


sea = (
    gam.groupby(["Season", "TeamID"])
    .agg(
        dict(
            **{
                c: "sum"
                for c in gam.drop(
                    columns=[
                        "Season",
                        "TeamID",
                        "OppID",
                        "DayNum",
                    ]
                )
            },
            DayNum="count",
        )
    )
    .reset_index()
)

sea["DayNum"] = sea["DayNum"].astype("int32")

sea = sea.rename(columns={c: f"{c[:-2]}s{c[-1]}" for c in sea if c[-3:-1] == "_g"})

sea = sea.rename(columns={"DayNum": "Games_su"})

print(f"sea {sea.shape}\n")
print(sea[[c for c in sea if c[-3:-1] != "_s"]])
print()
print(sea[[c for c in sea if c[-3:] == "_su"]])
print()
print(sea[[c for c in sea if c[-3:] == "_st"]])


# In[ ]:


gsx = pd.merge(gam, sea, on=["Season", "TeamID"])

gsx = gsx.rename(columns={c: f"T_{c}" for c in gsx if c[-3:-1] == "_s"})

gsx = pd.merge(
    gsx,
    sea,
    left_on=["Season", "OppID"],
    right_on=["Season", "TeamID"],
    suffixes=["", "_fromseason"],
)

gsx = gsx.drop(columns="TeamID_fromseason")

gsx = gsx.rename(
    columns={c: f"O_{c}" for c in gsx if c[:2] != "T_" and c[-3:-1] == "_s"}
)

print(f"gsx {gsx.shape}\n")
print(gsx[[c for c in gsx if c[-3:-1] not in ("_g", "_s")]])
print()
print(gsx[[c for c in gsx if c[-3:] == "_gu"]])
print()
print(gsx[[c for c in gsx if c[-3:] == "_gt"]])
print()
print(gsx[[c for c in gsx if c[:2] == "T_" and c[-3:] == "_su"]])
print()
print(gsx[[c for c in gsx if c[:2] == "T_" and c[-3:] == "_st"]])
print()
print(gsx[[c for c in gsx if c[:2] == "O_" and c[-3:] == "_su"]])
print()
print(gsx[[c for c in gsx if c[:2] == "O_" and c[-3:] == "_st"]])


# In[ ]:


aop = (
    gsx.groupby(["Season", "TeamID"])[[c for c in gsx if c[:2] == "O_"]]
    .sum()
    .reset_index()
)

aop = aop.rename(columns={c: f"{c[2:-2]}a{c[-1]}" for c in aop if c[-3:-1] == "_s"})

print(f"aop {aop.shape}\n")
print(aop[[c for c in aop if c[-3:-1] != "_a"]])
print()
print(aop[[c for c in aop if c[-3:] == "_au"]])
print()
print(aop[[c for c in aop if c[-3:] == "_at"]])


# In[ ]:


gsa = pd.merge(gsx, aop, on=["Season", "TeamID"])

stats = [c[:-3] for c in gam if c[-3:] == "_gu" and c not in ("Loc_gu", "NumOT_gu")]

for stat in stats:
    gsa[f"T_{stat}_su"] = gsa[f"T_{stat}_su"] - gsa[f"{stat}_gu"]
    gsa[f"T_{stat}_st"] = gsa[f"T_{stat}_st"] - gsa[f"{stat}_gt"]

    gsa[f"O_{stat}_su"] = gsa[f"O_{stat}_su"] - gsa[f"{stat}_gt"]
    gsa[f"O_{stat}_st"] = gsa[f"O_{stat}_st"] - gsa[f"{stat}_gu"]

    gsa[f"{stat}_au"] = gsa[f"{stat}_au"] - gsa[f"{stat}_gt"]
    gsa[f"{stat}_at"] = gsa[f"{stat}_at"] - gsa[f"{stat}_gu"]

for stat in ["Loc", "NumOT"]:
    gsa[f"T_{stat}_su"] = gsa[f"T_{stat}_su"] - gsa[f"{stat}_gu"]
    gsa[f"O_{stat}_su"] = gsa[f"O_{stat}_su"] + gsa[f"{stat}_gu"]
    gsa[f"{stat}_au"] = gsa[f"{stat}_au"] + gsa[f"{stat}_gu"]

gsa[f"T_Games_su"] -= 1
gsa[f"O_Games_su"] -= 1
gsa[f"Games_au"] -= 1

gsa = gsa.drop(columns=[c for c in gsa if c[-3:-1] == "_g"])

print(f"gsa {gsa.shape}\n")
print(gsa[[c for c in gsa if c[-3:-1] not in ("_s", "_a")]])
print()
print(gsa[[c for c in gsa if c[:2] == "T_" and c[-3:] == "_su"]])
print()
print(gsa[[c for c in gsa if c[:2] == "T_" and c[-3:] == "_st"]])
print()
print(gsa[[c for c in gsa if c[:2] == "O_" and c[-3:] == "_su"]])
print()
print(gsa[[c for c in gsa if c[:2] == "O_" and c[-3:] == "_st"]])
print()
print(gsa[[c for c in gsa if c[-3:] == "_au"]])
print()
print(gsa[[c for c in gsa if c[-3:] == "_at"]])


# In[ ]:


pg = gsa.copy()

for stat in stats:
    pg[f"T_{stat}_su"] = (pg[f"T_{stat}_su"] / pg[f"T_Games_su"]).astype("float32")
    pg[f"T_{stat}_st"] = (pg[f"T_{stat}_st"] / pg[f"T_Games_su"]).astype("float32")

    pg[f"O_{stat}_su"] = (pg[f"O_{stat}_su"] / pg[f"O_Games_su"]).astype("float32")
    pg[f"O_{stat}_st"] = (pg[f"O_{stat}_st"] / pg[f"O_Games_su"]).astype("float32")

    pg[f"{stat}_au"] = (pg[f"{stat}_au"] / pg[f"Games_au"]).astype("float32")
    pg[f"{stat}_at"] = (pg[f"{stat}_at"] / pg[f"Games_au"]).astype("float32")

for stat in ["Loc", "NumOT"]:
    pg[f"T_{stat}_su"] = (pg[f"T_{stat}_su"] / pg[f"T_Games_su"]).astype("float32")
    pg[f"O_{stat}_su"] = (pg[f"O_{stat}_su"] / pg[f"O_Games_su"]).astype("float32")
    pg[f"{stat}_au"] = (pg[f"{stat}_au"] / pg[f"Games_au"]).astype("float32")

pg = pg.drop(columns=["T_Games_su", "O_Games_su", "Games_au"])

print(f"pg {pg.shape}\n")
print(pg[[c for c in pg if c[-3:-1] not in ("_s", "_a")]])
print()
print(pg[[c for c in pg if c[:2] == "T_" and c[-3:] == "_su"]])
print()
print(pg[[c for c in pg if c[:2] == "T_" and c[-3:] == "_st"]])
print()
print(pg[[c for c in pg if c[:2] == "O_" and c[-3:] == "_su"]])
print()
print(pg[[c for c in pg if c[:2] == "O_" and c[-3:] == "_st"]])
print()
print(pg[[c for c in pg if c[-3:] == "_au"]])
print()
print(pg[[c for c in pg if c[-3:] == "_at"]])


# In[ ]:


train = pd.merge(tar, pg, on=["Season", "DayNum", "TeamID"], suffixes=["", "_"])
train = train.drop(columns="OppID_")
train.to_csv("train_daynum.csv", index=False)
print(f"train {train.shape}")
print(train)


# In[ ]:




