import torch 
from torch.nn.utils.rnn import pad_sequence

def combine_batch(batch):
    keys, X, Y, seq_lens = zip(*batch)
    max_seq = max(seq_lens)

    # batch of features of size (batch, timesteps, num_features)
    X = pad_sequence(X, batch_first=True)

    # batch of labels of size (batch, num_players: 11)
    #Y = [y.unsqueeze(0).repeat(seq_len, 1) for y, seq_len in zip(Y, seq_lens)]
    Y = pad_sequence(Y, padding_value=-1, batch_first=True)

    return keys, X, Y, seq_lens

def combine_batch2(batch):
    keys, X, Z, Y = zip(*batch)
    X = torch.stack(X, 0)
    Z = torch.stack(Z, 0)
    Y = torch.stack(Y, 0)
    return keys, X, Z, Y

def create_src_key_padding_mask(sequence_lengths: list, max_length: int=None):

    # Determine the batch size and max length
    batch_size = len(sequence_lengths)
    if max_length is None:
        max_length = max(sequence_lengths)
    

    mask = torch.ones((batch_size, max_length), dtype=torch.bool)    
    # Set False for non-padding positions
    for i, seq_len in enumerate(sequence_lengths):
        mask[i, :seq_len] = False 

    return mask