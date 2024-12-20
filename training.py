import numpy as np
import pandas as pd
import torch 

def get_lstm_output(model: torch.nn.Module, 
                    data_loader: torch.utils.data.DataLoader,
                    return_hidden_states: bool=False):

    #all_data = BlitzData(train + val)
    #all_data_loader = DataLoader(all_data, batch_size=128, collate_fn=combine_batch)
    all_df = pd.DataFrame()
    model.eval()
    for i, (keys, X, Y, seq_lens) in enumerate(data_loader):

        with torch.no_grad():
            out = model(X, seq_lens, return_hidden_states=return_hidden_states).numpy()
            model_dim = out.shape[2] if return_hidden_states else 1


        cols = ["game_id", "play_id", "player_id"]
        cols += [f"lstm{i}" for i in range(model_dim)]

        for k in range(out.shape[0]):
            if model_dim > 1:
                x = out[k, :, :].squeeze() # (T, M/2)
                x = x[:seq_lens[k], :]
            else:
                x = out[k, :].reshape(-1, 1)
                x = x[:seq_lens[k], :]

            x_keys = np.repeat(keys[k], x.shape[0]).reshape(3, -1).T
            x = np.concatenate([x_keys, x], axis=1)

            df = pd.DataFrame(x, columns=cols)
            all_df = pd.concat([all_df, df])

    all_df = all_df.sort_values(["game_id", "play_id", "player_id"]).reset_index(drop=True)
    all_df["index"] = all_df.groupby(["game_id", "play_id", "player_id"]).cumcount()

    return all_df

def tidy_val_preds(preds: dict, data: pd.DataFrame, label:str="y"):

    indices = []
    values = []
    for k, v in preds.items():
        indices.append(k)
        values.append(v)

    val_preds_df = pd.DataFrame(np.stack(values), 
                                columns=[str(x) for x in np.arange(11)], 
                                index=pd.MultiIndex.from_tuples(indices, names=["game_id", "play_id", "frame_id"]))
    val_preds_df = val_preds_df.reset_index()
    val_preds_df = pd.melt(val_preds_df, 
                        id_vars=["game_id", "play_id", "frame_id"],
                        var_name="player_id",
                        value_name="pred")
    
    val_preds_df["player_id"] = val_preds_df["player_id"].astype("int")
    val_preds_df = val_preds_df.merge(data[["game_id", "play_id", "frame_id", "player_id", label]], 
                                      on=["game_id", "play_id", "frame_id", "player_id"])
    val_preds_df = val_preds_df.reset_index(drop=True)

    val_preds_df["time_to_snap"] = (
    val_preds_df
    .groupby(["game_id", "play_id"])["frame_id"]
    .transform(lambda x: x.max() - x)
    )
    val_preds_df = val_preds_df.loc[val_preds_df[label] != -1].reset_index(drop=True)
    val_preds_df["pred_max"] = (
        val_preds_df
        .groupby(["game_id", "play_id", "player_id"])["pred"]
        .transform(lambda x: x.cummax())
        .reset_index(drop=True)
    )

    return val_preds_df


def get_transformer_frame_embeddings(data_loader: torch.utils.data.DataLoader,
                                     model: torch.nn.Module):
    
    model.eval()
    data = []
    for keys, X, Z, Y in data_loader:

        with torch.no_grad():
            # dim = (B x 11 x D)
            embs = model(X, Z, return_embeddings=True)
        reshaped_output = embs.reshape(-1, embs.shape[2]).numpy() 

        # Repeat keys for each of the 11 items in the second dimension
        expanded_keys = np.repeat(keys, 11, axis=0)

        # Generate ids (0 to 10) for the second dimension and tile them across batches
        ids = np.tile(np.arange(11), X.shape[0])

        # Combine keys and ids into a multi-level index with four levels
        index = pd.MultiIndex.from_tuples(
            [(game_id, play_id, player_id, id_) for (game_id, play_id, player_id), id_ in zip(expanded_keys, ids)],
            names=["game_id", "play_id", "frame_id", "player_id"]
        )

        df = pd.DataFrame(reshaped_output, index=index)
        data.append(df)

    return pd.concat(data) 

def train_epoch2(data_loader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: callable,
                 device: torch.device,
                 pool: bool=False):
    
    running_loss = 0
    n_batches = 0
    model.train()

    data = []
    for _, X, Z, Y in data_loader:
        X, Z, Y = X.to(device), Z.to(device), Y.to(device)

        model.zero_grad()
        out = model(X, Z)

        if pool:
            # out = out * (Y != -1)
            # out = out.sum(dim=1)
            # out = torch.clamp_max(out, 1.)
            Y = (Y == 1).any(dim=1).float()
        else:
            mask = (Y != -1).flatten()
            Y = Y.flatten()[mask]
            out = out.flatten()[mask]

        data.append(torch.stack((out, Y), dim=1).detach().cpu().numpy())

        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    epoch_loss = running_loss / n_batches

    return epoch_loss, data

def validate_epoch2(data_loader: torch.utils.data.DataLoader,
                    model: torch.nn.Module,
                    loss_fn: callable,
                    device: torch.device,
                    pool: bool=False):
    
        model.eval()
        running_loss = 0
        n_batches = 0

        data = {}
        for keys, X, Z, Y in data_loader:
            X, Z, Y = X.to(device), Z.to(device), Y.to(device)

            with torch.no_grad():
                out = model(X, Z)
            
            # save preds with game, play, and frame ids
            for i, key in enumerate(keys):
                data[key] = out[i].cpu().numpy()
            
            if pool:
                # sum individual pass rusher pressure probs to assess at the play level 
                # out = out * (Y != -1)
                # out = out.sum(dim=1)
                # out = torch.clamp_max(out, 1.)
                Y = (Y == 1).any(dim=1).float()
            else:
                mask = (Y != -1).flatten()
                Y = Y.flatten()[mask]
                out = out.flatten()[mask]

            loss = loss_fn(out, Y).item()
            running_loss += loss
            n_batches += 1
        
        epoch_loss = running_loss / n_batches 

        return epoch_loss, data 


def train_epoch(data_loader: torch.utils.data.DataLoader,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                loss_fn: callable,
                final_output: bool=False):
    
    running_loss = 0
    n_batches = 0
    model.train()

    for i, (_, X, Y, seq_lens) in enumerate(data_loader):

        model.zero_grad()
        out = model(X, seq_lens)

        if final_output:
            col_indices = [x - 1 for x in seq_lens]
            out = out[torch.arange(out.shape[0]), col_indices].squeeze()
            Y = Y[:, 0]
        else:
            # create label mask so padded time steps are ignored in the loss calc 
            mask = (Y != -1).flatten()
            Y = Y.flatten()[mask]
            out = out.flatten()[mask]

        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    epoch_loss = running_loss / n_batches

    return epoch_loss 

def validate_epoch(data_loader: torch.utils.data.DataLoader,
                   model: torch.nn.Module,
                   loss_fn: callable,
                   final_output: bool=True):
    
    model.eval()
    running_loss = 0
    n_batches = 0

    data = {}
    for keys, X, Y, seq_lens in data_loader:

        with torch.no_grad():
            out = model(X, seq_lens)
        
        # handle batch size 1 
        if len(out.size()) == 1:
            out = out.unsqueeze(0)

        # save preds to combine later 
        for i, key in enumerate(keys):
            n_rows = seq_lens[i]
            data[key] = out[i, :n_rows].numpy()

        if final_output:
            col_indices = [x - 1 for x in seq_lens]
            out = out[torch.arange(out.shape[0]), col_indices].squeeze()
            Y = Y[:, 0]
            if len(Y) == 1:
                Y = Y.squeeze()
        else:
            # create label mask so padded time steps are ignored in the loss calc 
            mask = (Y != -1).flatten()
            Y = Y.flatten()[mask]
            out = out.flatten()[mask]

        loss = loss_fn(out, Y).item()

        running_loss += loss 
        n_batches += 1

    epoch_loss = running_loss / n_batches 

    return epoch_loss, data
        