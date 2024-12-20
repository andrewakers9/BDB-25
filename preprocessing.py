import numpy as np
import pandas as pd

def get_cv_val_ids(player_data: pd.DataFrame):
    val_ids = []
    for i in range(player_data["week"].nunique()):
        val_ids.append(player_data.loc[player_data["week"] == i + 1].index)
    return val_ids

def get_data_splits(player_data: pd.DataFrame, 
                    play_data: dict, 
                    val_ids: pd.Index,
                    mirror_train: bool=False,
                    feats: list=["rel_x", "rel_y", "speed_x", "speed_y", "position_id"],
                    label: str="y"):

    train_df = player_data.loc[~player_data.index.isin(val_ids)]
    val_df = player_data.loc[player_data.index.isin(val_ids)]

    train = []
    for key, df in train_df.groupby(["game_id", "play_id", "frame_id"]):
        if len(df) != 22:
            continue
        y = df[label].values[:11]
        X = df[feats].values
        z = play_data[key[:2]]
        train.append((key, X, z, y))
        if mirror_train:
            df[["rel_y", "speed_y"]] *= -1
            X_mirror = df[feats].values
            train.append((key, X_mirror, z, y))
    val = []
    for key, df in val_df.groupby(["game_id", "play_id", "frame_id"]):
        if len(df) != 22:
            continue
        y = df[label].values[:11]
        X = df[feats].values
        z = play_data[key[:2]]
        val.append((key, X, z, y))

    return train, val