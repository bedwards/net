#!/usr/bin/env python

# import warnings

# warnings.simplefilter("ignore")

import os
import numpy as np
import pandas as pd

is_notebook = False
p = print
data_dir = f"../datasets/march-machine-learning-mania-2025"


def read_detailed_results(fn):
    df = pd.read_csv(f"{data_dir}/{fn}.csv")
    for c in df.select_dtypes("int64"):
        df[c] = df[c].astype("int32")
    df = df.rename(columns={"Season": "WSeason"})
    df["LSeason"] = df["WSeason"]
    df = df.rename(columns={"DayNum": "WDayNum"})
    df["LDayNum"] = df["WDayNum"]
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

c_not_WL = [c for c in detailed_results if c[0] not in ("W", "L")]
assert 0 == len(c_not_WL), c_not_WL
print(f"\n{[c[1:] for c in detailed_results if c[0] == 'W']}\n")

margin = detailed_results[
    ["WSeason", "WDayNum", "WTeamID", "LTeamID", "WScore", "LScore"]
]
margin = margin.rename(columns={"WSeason": "Season", "WDayNum": "DayNum"})
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
margin["Margin"] = margin["Margin"].astype("int32")

margin = margin.drop(columns=["WTeamID", "LTeamID", "WScore", "LScore"])

margin['game_key'] = margin['Season'].astype(str) + "_" + \
                     margin['DayNum'].astype(str) + "_" + \
                     margin['TeamID_1'].astype(str) + "_" + \
                     margin['TeamID_2'].astype(str)

p(margin)
margin.info()
print()

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

game_team = game_team.rename(columns={"Season_o": "Season", "DayNum_o": "DayNum", "TeamID_o": "TeamID"})
game_team = game_team.drop(columns=["Season_d", "DayNum_d", "TeamID_d"])
game_team = calculate_possessions(game_team)

game_team = game_team[
    ["Season", "DayNum", "TeamID", "Opponent"]
    + [c for c in game_team if c[-2:] == "_o"]
    + [c for c in game_team if c[-2:] == "_d"]
]

game_team['game_key'] = game_team['Season'].astype(str) + "_" + \
                        game_team['DayNum'].astype(str) + "_" + \
                        np.minimum(game_team['TeamID'], game_team['Opponent']).astype(str) + "_" + \
                        np.maximum(game_team['TeamID'], game_team['Opponent']).astype(str)

p(game_team)

# Create a copy of game_team with renamed columns for merging
game_stats = game_team.copy()
game_stats = game_stats.rename(columns={c: f"{c}_game" for c in game_stats if c[-2] == "_"})


# season_sum = (
#     game_team.drop(columns="Opponent")
#     .groupby(["Season", "TeamID"])
#     .agg(
#         dict(
#             Score_o=["count", "sum"],
#             **{c: "sum" for c in game_team if c[-2] == "_" and c != "Score_o"},
#         )
#     )
# )
# season_sum.columns = season_sum.columns.map("_".join)
# season_sum = season_sum.rename(columns={"Score_o_count": "Games"})

season_sum = (
    game_team.drop(columns="Opponent")
    .groupby(["Season", "TeamID"])
    .agg(
        {c: "sum" for c in game_team if c[-2] == "_"},
    )
)
season_sum = season_sum.rename(columns=
    {c: f"{c}_sum" for c in game_team if c[-2] == "_"},
)

season_sum = season_sum.reset_index()
p(season_sum)

def from_sum_to_pct(sum_df, prefix=""):
    pct_df = sum_df.copy()

    for c in pct_df:
        if c.endswith("_sum"):
            # pct_df[f"{c[:-6]}_pg_{c[-5:-4]}"] = (pct_df[c] / pct_df["Games"]).astype(
            #     "float32"
            # )
            if "_o_" in c:
                if c == "Poss_o":
                    print("YYYYYYYYYEEEEEEEEEESSSSS!!!!")
                pct_df[f"{c[:-6]}_poss_{c[-5:-4]}"] = (
                    pct_df[c] / pct_df[f"{prefix}Poss_o_sum"]
                ).astype("float32")
            if "_d_" in c:
                pct_df[f"{c[:-6]}_poss_{c[-5:-4]}"] = (
                    pct_df[c] / pct_df[f"{prefix}Poss_d_sum"]
                ).astype("float32")

    # for side in ["o", "d"]:
    #     pct_df[f"{prefix}FGPct_{side}"] = (
    #         pct_df[f"{prefix}FGM_{side}_sum"] / pct_df[f"{prefix}FGA_{side}_sum"]
    #     ).astype("float32")
    #     pct_df[f"{prefix}FGPct3_{side}"] = (
    #         pct_df[f"{prefix}FGM3_{side}_sum"] / pct_df[f"{prefix}FGA3_{side}_sum"]
    #     ).astype("float32")
    #     pct_df[f"{prefix}FTPct_{side}"] = (
    #         pct_df[f"{prefix}FTM_{side}_sum"] / pct_df[f"{prefix}FTA_{side}_sum"]
    #     ).astype("float32")
    #     pct_df[f"{prefix}AstPct_{side}"] = (
    #         pct_df[f"{prefix}Ast_{side}_sum"] / pct_df[f"{prefix}FGM_{side}_sum"]
    #     ).astype("float32")
    #     pct_df[f"{prefix}AstTO_{side}"] = (
    #         pct_df[f"{prefix}Ast_{side}_sum"] / pct_df[f"{prefix}TO_{side}_sum"]
    #     ).astype("float32")

    pct_df = pct_df.drop(columns=[c for c in pct_df if c.endswith("_sum")])

    # pct_df = pct_df.drop(
    #     columns=["Games", f"{prefix}Poss_poss_o", f"{prefix}Poss_poss_d"]
    # )
    pct_df = pct_df.drop(
        columns=[f"{prefix}Poss_poss_o", f"{prefix}Poss_poss_d"]
    )

    # for c in pct_df:
    #     if c.endswith("_o"):
    #         pct_df[f"{c[:-2]}_diff"] = pct_df[c] - pct_df[f"{c[:-2]}_d"]

    return pct_df


season_team = from_sum_to_pct(season_sum)
p(season_team)
season_team.info()
print(flush=True)

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


sum_sos = (
    game_sos.drop(columns="Opponent")
    .groupby(["Season", "TeamID"], as_index=False)
    .sum()
)
p(sum_sos[(sum_sos["Season"] > 2021) & (sum_sos["TeamID"] == 1196)])


season_sos = from_sum_to_pct(sum_sos, prefix="sos_")
p(season_sos[(sum_sos["Season"] > 2021) & (season_sos["TeamID"] == 1196)])


season = pd.merge(season_team, season_sos, on=["Season", "TeamID"])
print("Writing season_2025.csv")
season[season["Season"] == 2025].to_csv("season_2025.csv", index=False)
p(season)


train = pd.merge(
    margin, season, left_on=["Season", "TeamID_1"], right_on=["Season", "TeamID"]
)
train = train.drop(columns="TeamID")

# [BEGIN NEW CODE] ------------------------------------------------------------

# Then merge with the specific game stats for TeamID_1
train = pd.merge(
    train,
    game_stats,
    left_on=["game_key", "Season", "DayNum", "TeamID_1"],
    right_on=["game_key", "Season", "DayNum", "TeamID"]
)
train = train.drop(columns=["TeamID", "Opponent"])

# Adjust the season sum columns by subtracting the game values
for col in [c for c in train.columns if "_sum" in c]:
    base_col = col.replace("_sum", "")
    game_col = f"{base_col}_game"
    if game_col in train.columns:
        train[col] = train[col] - train[game_col]

# Drop the game columns
train = train.drop(columns=[c for c in train.columns if "_game" in c])

# [END NEW CODE] --------------------------------------------------------------


train = train.rename(
    columns={c: f"{c}_1" for c in train if c.split("_")[-1] in ("o", "d", "diff")}
)

train = pd.merge(
    train, season, left_on=["Season", "TeamID_2"], right_on=["Season", "TeamID"]
)
train = train.drop(columns="TeamID")


# [BEGIN NEW CODE] ------------------------------------------------------------

# Merge with game stats for TeamID_2
train = pd.merge(
    train,
    game_stats,
    left_on=["game_key", "Season", "DayNum", "TeamID_2"],
    right_on=["game_key", "Season", "DayNum", "TeamID"]
)
train = train.drop(columns=["TeamID", "Opponent"])

# Adjust season sum columns for TeamID_2
for col in [c for c in train.columns if "_sum" in c and not col.endswith("_1")]:
    base_col = col.replace("_sum", "")
    game_col = f"{base_col}_game"
    if game_col in train.columns:
        train[col] = train[col] - train[game_col]

# Drop the game columns
train = train.drop(columns=[c for c in train.columns if "_game" in c])

# [END NEW CODE] --------------------------------------------------------------

train = train.rename(
    columns={c: f"{c}_2" for c in train if c.split("_")[-1] in ("o", "d", "diff")}
)

train = train.sort_values(["Season", "DayNum", "TeamID_1", "TeamID_2"]).reset_index(
    drop=True
)

p(train)
train.info()
print()
print(train.select_dtypes("float64").columns, flush=True)

fn = "train_poss.csv"
print(f"Writing {fn}")
train.to_csv(f"{fn}", index=False)
