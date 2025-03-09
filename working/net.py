#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    get_ipython().run_line_magic("reset", "-f")
except NameError:
    is_notebook = False
    p = print
else:
    is_notebook = True
    p = display
    # p = print

import warnings

warnings.simplefilter("ignore")

import pandas as pd

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

if is_notebook:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["figure.figsize"] = (10, 2)


# In[2]:


data_dir = f"../datasets/march-machine-learning-mania-2025"


# In[3]:


def read_detailed_results(fn):
    df = pd.read_csv(f"{data_dir}/{fn}.csv")
    for c in df.select_dtypes("int64"):
        df[c] = df[c].astype("int32")
    df = df.rename(columns={"Season": "WSeason"})
    df["LSeason"] = df["WSeason"]
    df = df.drop(columns=["DayNum", "NumOT", "WLoc"])
    df.index = df.index.astype("int32")
    assert all(df[c].dtype == "int32" for c in df)
    assert df.index.dtype == "int32"
    print(
        f"{fn:<29} {df.shape[0]:>7,} {df.shape[1]:>2} {df['WSeason'].min()} {df['WSeason'].max()}"
    )
    return df


MNCAATourneyDetailedResults = read_detailed_results("MNCAATourneyDetailedResults")
MRegularSeasonDetailedResults = read_detailed_results("MRegularSeasonDetailedResults")
WNCAATourneyDetailedResults = read_detailed_results("WNCAATourneyDetailedResults")
WRegularSeasonDetailedResults = read_detailed_results("WRegularSeasonDetailedResults")

detailed_results = pd.concat(
    [
        MNCAATourneyDetailedResults,
        MRegularSeasonDetailedResults,
        WNCAATourneyDetailedResults,
        WRegularSeasonDetailedResults,
    ]
).reset_index(drop=True)

print(
    f"{'-'*50}\n{'detailed_results':<29} {detailed_results.shape[0]:>7,} {detailed_results.shape[1]:>2} {detailed_results['WSeason'].min()} {detailed_results['WSeason'].max()}"
)
assert 0 == len([c for c in detailed_results if c[0] not in ("W", "L")])
print(f"\n{[c[1:] for c in detailed_results if c[0] == 'W']}")

if is_notebook:
    sns.histplot(detailed_results["WSeason"])


# In[4]:


def drop_WL(W_or_L):
    df = detailed_results[[c for c in detailed_results if c[0] == W_or_L]]
    return df.rename(columns={c: c[1:] for c in df})


game_team = pd.concat([drop_WL("W"), drop_WL("L")]).reset_index(drop=True)
p(game_team)

if is_notebook:
    sns.barplot(
        game_team.groupby("Season")["TeamID"].nunique().reset_index(name="Teams"),
        x="Season",
        y="Teams",
    )

    plt.xticks(rotation=90)
    plt.figure()

    sns.barplot(
        game_team.groupby(["Season", "TeamID"])["Score"]
        .count()
        .reset_index()
        .groupby("Season")["Score"]
        .median()
        .reset_index(name="Median games per team"),
        x="Season",
        y="Median games per team",
    )

    plt.xticks(rotation=90)


# In[ ]:
