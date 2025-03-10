#!/usr/bin/env python
# coding: utf-8

# This notebook offers preprocessing of the [March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) Kaggle competition. The code in this notebook targets point margin instead of win probability. A point-margin predicting model using the data produced by this notebook can convert its point-margin predictions to win probability. One way to do this is `1 / (1 + np.exp(-margin * 0.25))` -- [numpy.exp](https://numpy.org/doc/stable/reference/generated/numpy.exp.html).
#
# From claude.ai:
#
# > This formula is the logistic function, and its use for converting point margins to win probabilities is often called the "logistic method" or "logistic regression approach" to margin-based win probability. The coefficient (0.25) represents how much a 1-point change in margin affects win probability. This specific value means that a 4-point favorite has approximately a 75% chance of winning.
#
# I chose to target point margin instead of win probability directly because point margin is a continous floating point number with much richer information to train a model on than a boolean win/loss target. When comparing two undefeated teams that played the same level of competition, the one that won all its games by 20 points is better than the one that won its games by 1 point.

# #### The first cell
#
# I like to use Jupyter Lab `File > Save and Export Notebook As > Executable Script` to run my notebook code in a terminal and get a [nice diff in git](https://github.com/bedwards/net/commit/420415083f126cf86691b05635a691cd284f6264). I run `black .; ./net.py | tee net.txt` and commit the output too for a nice log of what changed. I format the notebook and script in one shot by installing `black .; ./net.py | tee net.txt`.
#
# `display` is the rich HTML formating of pandas DataFrame in notebooks. It is not available in the terminal so I create an alias `p` and switch between `display` and `print` depending on the environment.

# In[16]:


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

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if is_notebook:
    import matplotlib.pyplot as plt
    import seaborn as sns

    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    plt.rcParams["figure.figsize"] = (10, 2)
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})


# I use Kaggle and [Paperspace Gradient](https://www.paperspace.com/artificial-intelligence) (part of DigitalOcean.) Kaggle uses an `input` directory for datasets (CSV files), Paperspace uses `datasets`.

# In[17]:


data_dir = f"../datasets/march-machine-learning-mania-2025"
if not os.path.isdir(data_dir):
    data_dir = f"../input/march-machine-learning-mania-2025"


# This notebooks preprocessing exclusively uses the detailed result CSV files from the competition dataset. Both Women's and Men's. Both regular season and tournament results. It presents targets of both regular season and tournament game point margins. There is data leakage in the regular season targets because that game's statistics are part of (approximately 1/30th of) the features used to predict the margin. I will not have the future tournament game stats in the features when I do my final prediction, therefore "data leakage" which can give inflated cross-validation scores. I accept that trade off because if I was to remove the game being targeted from the season stats, that is a row-by-row operation. It cannot be [vectorized](https://pythonspeed.com/articles/pandas-vectorization/), and therefore is less efficient.

# In[18]:


def read_detailed_results(fn):
    df = pd.read_csv(f"{data_dir}/{fn}.csv")
    for c in df.select_dtypes("int64"):
        df[c] = df[c].astype("int32")
    df = df.rename(columns={"Season": "WSeason"})
    df["LSeason"] = df["WSeason"]
    df = df.drop(columns=["NumOT", "WLoc"])
    assert all(df[c].dtype == "int32" for c in df)
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
assert 1 == len([c for c in detailed_results if c[0] not in ("W", "L")])
print(f"\n{[c[1:] for c in detailed_results if c[0] == 'W']}\n")

if is_notebook:
    sns.histplot(detailed_results["WSeason"])
    plt.title(
        "Detailed results (before 2010: men only, 2021: COVID, 2025: regular season only)"
    )
    plt.xlabel("Season")
    plt.ylabel("Games")


# The data in the detailed results CSV files is centered around winning team statistics and losing team statistics. The competition submission is centered around lower TeamID statistics and higher TeamID statistics. Furthermore, presenting a model with season stats with separate columns for the winning team and the losing team would be an unacceptable amount of data leakage. The model would quickly recognize and become dependent on the win/loss encoded in the order of those columns, which will not be available when predicting the outcome of future tournament games.
#
# So we need to get from winner/loser to a more arbitrary and competition-submission aligned Team_1 and Team_2 (ordered by their competition-assigned ID numbers which are assigned in alphabetic order of team name in MTeams.csv and WTeams.csv - Men's are in the 1000s, Women's are in the 3000s.)
#
# #### The y
#
# I use the convention of naming the series of training target data `y`. The corresponding feature data is kept in a DataFrame named `X`. I have a `train` DataFrame (and this notebook produces a `train.csv` file) that includes the fields required to uniquely identify the rows (`Season`, `DayNum`, `TeamID_1` and `TeamID_2`) and three of those are used to construct and merge with the competition submission ID column in `SampleSubmissionStage2.csv`. The `train` DataFrame also include the `y` series and the `X` DataFrame.
#
# There is much more involved in preproccessing to produce the `X` DataFrame, so I begin with the simpley `y` series and basically produce it here. The `margin` DataFrame includes both the uniquely identifying rows and the `y` series.
#
# I use `int32` and `float32` which are important for saving GPU memory, but be warned that the `train.csv` file loses this information. Loading it with Pandas `read_csv` in another notebook will use the defaults for the environment which is most likely the more memory-consuming `int64` and `float64`.
#
# An aside, I will point out that I define one (maybe two or three) "major" variables per cell (variables that are intended to be referenced in later cells.) I mutate these variables only in the cell where they are declared. This allows me to confidently re-run a single cell in the middle of my notebook and expect repeatable results (because the cell or later cells aren't mutating their input.) I also place a `%reset -f` so running all cells is similar to restarting the notebook kernel, deleting all variables.

# In[19]:


margin = detailed_results[
    ["WSeason", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]
]
margin = margin.rename(columns={"WSeason": "Season"})
mask = margin["WTeamID"] < margin["LTeamID"]

margin.loc[mask, ["TeamID_1", "TeamID_2", "Margin"]] = np.column_stack(
    [
        margin.loc[mask, "WTeamID"].values,
        margin.loc[mask, "LTeamID"].values,
        (margin.loc[mask, "WScore"] - margin.loc[mask, "LScore"]).values,
    ]
)

margin.loc[~mask, ["TeamID_1", "TeamID_2", "Margin"]] = np.column_stack(
    [
        margin.loc[~mask, "LTeamID"].values,
        margin.loc[~mask, "WTeamID"].values,
        (margin.loc[~mask, "LScore"] - margin.loc[~mask, "WScore"]).values,
    ]
)

margin["TeamID_1"] = margin["TeamID_1"].astype("int32")
margin["TeamID_2"] = margin["TeamID_2"].astype("int32")
margin["Margin"] = margin["Margin"].astype("float32")
margin = margin.drop(columns=["WTeamID", "LTeamID", "WScore", "LScore"])

p(margin)
margin.info()
print()

if is_notebook:
    sns.lineplot(
        margin.groupby("Season")["Margin"].mean().reset_index(),
        x="Season",
        y="Margin",
    )
    plt.title(
        "Sanity check, average point margin should be near zero for arbitrary team perspective (lower TeamID)"
    )


# #### The beginnings of the X
#
# The features (`X`) will eventually be a quite-mind-bending assortment (about 400 columns) of
#
# - Stat lines that represent a single season
# - TeamID_1 and TeamID_2 stat lines
# - For each team:
#     - Stats per game, per possession, and team differentials of each
#     - In each of those categories the stats are sliced in various ways:
#         - own vs. dual
#             - "o" for "own" stats: Score, FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF
#             - "d" for "dual" stats: the same stats but for the "own" team's opponents in games played against the "own" team
#         - strength of schedule (sos)
#             - for all those combinations (team 1 and 2, per game, per poss, diff, own, dual)
#             - compute the strength of schedule
#             - which is the dual team's (opponent's) season stats against all their opponents
#             - weighted by number of time own and dual played each other during the season
#
# But first, own and dual game stats from each teams perspective.

# In[21]:


def from_WL_to_od(from_prefix, to_suffix):
    df = detailed_results[[c for c in detailed_results if c[0] == from_prefix]]
    return df.rename(columns={c: f"{c[1:]}_{to_suffix}" for c in df})


def calculate_possessions(df, prefix="", suffix=""):
    df["Poss_o"] = (
        df[f"{prefix}FGA_o{suffix}"]
        - df[f"{prefix}OR_o{suffix}"]
        + df[f"{prefix}TO_o{suffix}"]
        + 0.44 * df[f"{prefix}FTA_o{suffix}"]
    ).astype("float32")
    df["Poss_d"] = (
        df[f"{prefix}FGA_d{suffix}"]
        - df[f"{prefix}OR_d{suffix}"]
        + df[f"{prefix}TO_d{suffix}"]
        + 0.44 * df[f"{prefix}FTA_d{suffix}"]
    ).astype("float32")
    return df


game_team = pd.concat(
    [
        pd.concat(
            [
                from_WL_to_od("W", "o"),
                from_WL_to_od("L", "d"),
                detailed_results["LTeamID"].rename("Opponent"),
            ],
            axis=1,
        ),
        pd.concat(
            [
                from_WL_to_od("L", "o"),
                from_WL_to_od("W", "d"),
                detailed_results["WTeamID"].rename("Opponent"),
            ],
            axis=1,
        ),
    ]
).reset_index(drop=True)

game_team = game_team.rename(columns={"Season_o": "Season", "TeamID_o": "TeamID"})
game_team = game_team.drop(columns=["Season_d", "TeamID_d"])
game_team = calculate_possessions(game_team)

game_team = game_team[
    ["Season", "TeamID", "Opponent"]
    + [c for c in game_team if c[-2:] == "_o"]
    + [c for c in game_team if c[-2:] == "_d"]
]

p(game_team)

if is_notebook:
    sns.barplot(
        game_team.groupby("Season")["TeamID"].nunique().reset_index(name="Teams"),
        x="Season",
        y="Teams",
        color=sns.color_palette()[0],
    )

    plt.title("Sanity check, number of teams per year (before 2010: men only)")
    plt.xticks(rotation=90)
    plt.figure()

    sns.barplot(
        game_team.groupby(["Season", "TeamID"])["Score_o"]
        .count()
        .reset_index()
        .groupby("Season")["Score_o"]
        .median()
        .reset_index(name="Median games per team"),
        x="Season",
        y="Median games per team",
        color=sns.color_palette()[0],
    )

    plt.title(
        "Sanity check, games per team (upper 20s to 30, 2021: COVID, 2025: regular season only)"
    )
    plt.xticks(rotation=90)


# Sum them up for each team per season. Keep track of games played for calculating per-game stats later.

# In[23]:


season_sum = (
    game_team.drop(columns="Opponent")
    .groupby(["Season", "TeamID"])
    .agg(
        dict(
            Score_o=["count", "sum"],
            **{c: "sum" for c in game_team if c[-2] == "_" and c != "Score_o"},
        )
    )
)

season_sum.columns = season_sum.columns.map("_".join)
season_sum = season_sum.reset_index()
season_sum = season_sum.rename(columns={"Score_o_count": "Games"})
p(season_sum)


# Per game, per possession, and difference between own and dual of each.
#
# Some additional calculated stats
# - FGPct = FGM / FGA
# - FGPct3 = FGM3 / FGA3
# - FTPct = FTM / FTA
# - AstPct = Ast / FGM
# - AstTO = Ast / TO
#
# I am not too sure the differtial stats and these additional calculated stats help. They rub me the wrong way. The model should be able to learn these relationships (if they are important in predicting point margin.) I want the model to learn from first principles, I don't want to sway it with my preconceived notions of what makes a team win in a basketball game.
#
# On the other hand, we know with a high level of certainty that these relationships do exist between these stat columns and are fundamental to understanding why teams win games. We use our domain experise to give the model a head start in the right direction.
#
# I ran xgboost with and without these and didn't see much difference. I think the xgboost with them in scored worse, but it was not a very scientific experiment (other factors had changed.)

# In[24]:


def from_sum_to_pct(sum_df, prefix=""):
    pct_df = sum_df.copy()

    for c in pct_df:
        if c.endswith("_sum"):
            pct_df[f"{c[:-6]}_pg_{c[-5:-4]}"] = (pct_df[c] / pct_df["Games"]).astype(
                "float32"
            )
            if "_o_" in c:
                pct_df[f"{c[:-6]}_poss_{c[-5:-4]}"] = (
                    pct_df[c] / pct_df[f"{prefix}Poss_o_sum"]
                ).astype("float32")
            if "_d_" in c:
                pct_df[f"{c[:-6]}_poss_{c[-5:-4]}"] = (
                    pct_df[c] / pct_df[f"{prefix}Poss_d_sum"]
                ).astype("float32")

    for side in ["o", "d"]:
        pct_df[f"{prefix}FGPct_{side}"] = (
            pct_df[f"{prefix}FGM_{side}_sum"] / pct_df[f"{prefix}FGA_{side}_sum"]
        ).astype("float32")
        pct_df[f"{prefix}FGPct3_{side}"] = (
            pct_df[f"{prefix}FGM3_{side}_sum"] / pct_df[f"{prefix}FGA3_{side}_sum"]
        ).astype("float32")
        pct_df[f"{prefix}FTPct_{side}"] = (
            pct_df[f"{prefix}FTM_{side}_sum"] / pct_df[f"{prefix}FTA_{side}_sum"]
        ).astype("float32")
        pct_df[f"{prefix}AstPct_{side}"] = (
            pct_df[f"{prefix}Ast_{side}_sum"] / pct_df[f"{prefix}FGM_{side}_sum"]
        ).astype("float32")
        pct_df[f"{prefix}AstTO_{side}"] = (
            pct_df[f"{prefix}Ast_{side}_sum"] / pct_df[f"{prefix}TO_{side}_sum"]
        ).astype("float32")

    pct_df = pct_df.drop(columns=[c for c in pct_df if c.endswith("_sum")])
    pct_df = pct_df.drop(
        columns=["Games", f"{prefix}Poss_poss_o", f"{prefix}Poss_poss_d"]
    )

    for c in pct_df:
        if c.endswith("_o"):
            pct_df[f"{c[:-2]}_diff"] = pct_df[c] - pct_df[f"{c[:-2]}_d"]

    return pct_df


season_team = from_sum_to_pct(season_sum)
p(season_team)
season_team.info()
print(flush=True)

if is_notebook:
    sns.lineplot(
        season_team[season_team["TeamID"] == 1196],
        x="Season",
        y="Score_pg_o",
        label="Points per game",
    )
    sns.lineplot(
        season_team[season_team["TeamID"] == 1196],
        x="Season",
        y="Score_pg_d",
        label="Opponents' ppg",
    )
    plt.figure()
    sns.lineplot(
        season_team[season_team["TeamID"] == 1196],
        x="Season",
        y="Score_poss_o",
        label="Points per poss",
    )
    sns.lineplot(
        season_team[season_team["TeamID"] == 1196],
        x="Season",
        y="Score_poss_d",
        label="Opponents' ppp",
    )


# Team 1196 is the Florida Gators men's team and the first two colors of the seaborn pallete are orange and blue. Is it just coincidence or could it be the most important signal to predict on? ;-)
#
# #### Strength of schedule
#
# Now for some mindbending (it certainly became hard to name things in a clear and concise way.) As the print out says, these are the season stats (simple sums and possessions) for each opponent of TeamID (multiple rows for multiple matchups throughout the season.)
#
# So it is confusing because the "game" in `game_sos` is referring to one-game-per-row of the "own" team, but the rows of `game_sos` are season stats of the "dual" team. The season stats will need to be rolled up again in a weigthed average to create the season sos stats from the "own" team's perspective.

# In[26]:


game_sos = pd.merge(
    game_team[["Season", "TeamID", "Opponent"]],
    season_sum,
    left_on=["Season", "Opponent"],
    right_on=["Season", "TeamID"],
    suffixes=["", "_"],
)

game_sos = game_sos.drop(columns=["TeamID_"])
game_sos = game_sos.rename(
    columns={c: f"sos_{c}" for c in game_sos if c[-4:] == "_sum"}
)

print(
    "The season stats (simple sums and possessions) for each opponent of TeamID (multiple rows for multiple matchups throughout the season)"
)
p(game_sos)

if is_notebook:
    df = (
        game_sos.groupby(["Season", "TeamID", "Opponent"])["Games"]
        .count()
        .reset_index()
    )
    sns.barplot(
        df.groupby("Season")["Games"].max().reset_index(),
        x="Season",
        y="Games",
        color=sns.color_palette()[0],
    )
    plt.title("Sanity check, maximum occurences of same matchup within a season")
    plt.xticks(rotation=90)
    plt.show()
    df = df[df["Games"] == 6]
    df = df[df["TeamID"] < df["Opponent"]].reset_index(drop=True)
    mteams = pd.read_csv(f"{data_dir}/MTeams.csv")
    df["Team_1"] = pd.merge(df, mteams, left_on="TeamID", right_on="TeamID")["TeamName"]
    df["Team_2"] = pd.merge(df, mteams, left_on="Opponent", right_on="TeamID")[
        "TeamName"
    ]
    p(df[["Season", "Team_1", "Team_2", "Games"]])


# Six games against the same opponent! Could that be true, even in the COVID year?
#
# Yes, I looked it up: [goterriers opponent history](https://goterriers.com/sports/mens-basketball/opponent-history/holy-cross/102)
#
# ```
# 1 1/ 4/2021 W 83-76
# 2 1/ 5/2021 L 66-68
# 3 2/13/2021 L 65-82
# 4 2/14/2021 W 86-68
# 5 2/17/2021 W 78-69
# 6 2/24/2021 L 75-86
# ```

# The weighted sum I promised.

# In[27]:


sum_sos = (
    game_sos.drop(columns="Opponent")
    .groupby(["Season", "TeamID"], as_index=False)
    .sum()
)
p(sum_sos[(sum_sos["Season"] > 2021) & (sum_sos["TeamID"] == 1196)])


# And the season sos stats from the perspective of the "own" team. Promised and delivered.

# In[29]:


season_sos = from_sum_to_pct(sum_sos, prefix="sos_")
p(season_sos[(sum_sos["Season"] > 2021) & (season_sos["TeamID"] == 1196)])


# Putting it *all* (well not quite yet) together.
#
# 206 columns and I promised approximately 400. (Answer: each row needs the complete stat line of the mostly-arbitrarily assigned TeamID_1 and TeamID_2.)

# In[30]:


season = pd.merge(season_team, season_sos, on=["Season", "TeamID"])
print("Writing season_2025.csv")
season[season["Season"] == 2025].to_csv("season_2025.csv", index=False)
p(season)


# Hey, something called `train`, with two merges one on `TeamID_1` and one on `TeamID_2` and a `to_csv` call. We are at the finish line (of preprocessing.)

# In[31]:


train = pd.merge(
    margin, season, left_on=["Season", "TeamID_1"], right_on=["Season", "TeamID"]
)
train = train.drop(columns="TeamID")
train = train.rename(
    columns={c: f"{c}_1" for c in train if c.split("_")[-1] in ("o", "d", "diff")}
)

train = pd.merge(
    train, season, left_on=["Season", "TeamID_2"], right_on=["Season", "TeamID"]
)
train = train.drop(columns="TeamID")
train = train.rename(
    columns={c: f"{c}_2" for c in train if c.split("_")[-1] in ("o", "d", "diff")}
)

train = train.sort_values(["Season", "DayNum", "TeamID_1", "TeamID_2"]).reset_index(
    drop=True
)

print("Writing train.csv")
train.to_csv("train.csv", index=False)

# p(train)
train.info()
print()
print(train.select_dtypes("float64").columns, flush=True)


# Example of creating an `X` and a `y` from `train`.

# In[13]:


X_df = train.drop(["Season", "DayNum", "TeamID_1", "TeamID_2", "Margin"], axis=1)

print(X_df.columns)
print()
print(sorted(set(train.columns) - set(X_df.columns)))
print()

X = X_df.values
X = StandardScaler().fit_transform(X)

# keep y_scaler around to do inverse_transform
# before converting to win probability and Brier scoring
y_scaler = StandardScaler()
y = train["Margin"].values.reshape(-1, 1)
y = y_scaler.fit_transform(y)

print(
    f"{'detailed_results':<16} {detailed_results.shape[0]:>7,} {detailed_results.shape[1]:>3} {detailed_results['WSeason'].min()} {detailed_results['WSeason'].max()}"
)

print(
    f"{'train':<16} {train.shape[0]:>7,} {train.shape[1]:>3} {train['Season'].min()} {train['Season'].max()}"
)

print(f"{'X':<16} {X.shape[0]:>7,} {X.shape[1]:>3}")
print(f"{'y':<16} {y.shape[0]:>7,} {y.shape[1]:>3}", flush=True)
