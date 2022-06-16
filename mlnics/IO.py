"""
There is an error with error_analysis_fixed_net if we don't train the net first.
--Perhaps we also want to save data for the normalization
"""


import torch
import numpy as np
import os
import pickle
import glob

def save_state(epoch, ronn, data, optimizer, train_loss, val_loss, suffix=""):
    """
    Save the reduced order neural network state which includes
        - neural network weights
        -
        ...

    Creates several different files including
        - checkpoint_{epoch}.pt (for neural network weights, optimizer, etc.)
        - metadata_{epoch}.pkl (for storing training and validation losses and anything else we might want)
        - data_metadata.pkl (for storing info about how to obtain training and validation data)
    """

    folder = ronn.reduction_method.folder_prefix + "/checkpoints" + suffix

    # make folder for training checkpoints
    if not os.path.exists(folder):
        os.mkdir(folder)

    # save info for training and validation sets if not already saved
    data_folder = folder + "/data_metadata"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

        torch.save(data.train_idx, data_folder + "/train_idx.pt")
        torch.save(data.train_data, data_folder + "/train_data.pt")
        torch.save(data.val_idx, data_folder + "/val_idx.pt")
        torch.save(data.val_data, data_folder + "/val_data.pt")

        # save the non-Tensor attributes
        data_metadata = {
            "name": ronn.name,
            "validation_proportion": data.validation_proportion,
            "initialized": data.initialized
        }
        with open(data_folder + "/basic_attributes.pkl", 'wb') as f:
            pickle.dump(data_metadata, f)

    # save the model
    torch.save({
        'epoch': epoch,
        'model_state_dict': ronn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, folder + f"/checkpoint_{epoch}.pt")


    current_train_loss, train_loss_fn = train_loss
    current_val_loss, val_loss_fn = val_loss

    metadata = {
        'epoch': epoch,
        'train_loss': current_train_loss.item(),
        'train_loss_type': train_loss_fn.name()
    }

    if val_loss_fn is not None:
        metadata['validation_loss'] = current_val_loss.item()
        metadata['validation_loss_type'] = val_loss_fn.name()

    with open(folder + f"/metadata_{epoch}.pkl", 'wb') as f:
        pickle.dump(metadata, f)



def load_state(epoch, ronn, data, optimizer, suffix=""):
    folder = ronn.reduction_method.folder_prefix + "/checkpoints" + suffix

    # load the data attributes
    data_folder = folder + "/data_metadata"
    data.train_idx = torch.load(data_folder + "/train_idx.pt")
    data.train_data = torch.load(data_folder + "/train_data.pt")
    data.val_idx = torch.load(data_folder + "/val_idx.pt")
    data.val_data = torch.load(data_folder + "/val_data.pt")
    with open(data_folder + "/basic_attributes.pkl", 'rb') as f:
        data_metadata = pickle.load(f)
    data.validation_proportion = data_metadata["validation_proportion"]
    data.initialized = data_metadata["initialized"]
    ronn.name = data_metadata["name"]

    # load the model and optimizer parameters
    checkpoint = torch.load(folder + f"/checkpoint_{epoch}.pt")
    ronn.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ronn.train()


def choose_state(ronn, suffix="", by_validation=True):
    """
    Returns the epoch number to load by determining the epoch with the
    lowest validation loss.
    """
    if by_validation:
        loss_string = "validation_loss"
    else:
        loss_string = "train_loss"

    folder = ronn.reduction_method.folder_prefix + "/checkpoints" + suffix
    metadata_files = glob.glob(folder + "/metadata_*.pkl")
    losses, epochs = [], []
    for path in metadata_files:
        with open(path, 'rb') as f:
            metadata = pickle.load(f)
            loss = metadata[loss_string]
            epoch = metadata['epoch']
            losses.append(loss)
            epochs.append(epoch)

    loss, idx = min((val, idx) for (idx, val) in enumerate(losses))
    epoch = epochs[idx]
    return epoch


def read_losses(ronn, suffix=""):
    folder = ronn.reduction_method.folder_prefix + "/checkpoints" + suffix
    metadata_files = glob.glob(folder + "/metadata_*.pkl")
    losses, epochs = [], []
    val_losses = []
    for path in metadata_files:
        with open(path, 'rb') as f:
            metadata = pickle.load(f)
            loss = metadata["train_loss"]
            epoch = metadata['epoch']
            if "validation_loss" in metadata:
                val_loss = metadata["validation_loss"]
                val_losses.append(val_loss)
            losses.append(loss)
            epochs.append(epoch)

    idx = np.argsort(epochs)

    return np.array(epochs)[idx], np.array(losses)[idx], np.array(val_losses)[idx]
