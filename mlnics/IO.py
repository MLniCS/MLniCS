"""
There is an error with error_analysis_fixed_net if we don't train the net first.
--Perhaps we also want to save data for the normalization
"""


import torch
import numpy as np
import os
import pickle
import glob

NN_FOLDER = "/nn_results"

def save_state(epoch, ronn, data, optimizer, train_loss_fn, val_loss_fn, epochs, train_losses, validation_losses):
    """
    Save the reduced order neural network state which includes
        - neural network weights
        -
        ...

    Creates several different files including
        - checkpoint.pt (for neural network weights, optimizer, etc.)
        - metadata_{epoch}.pkl (for storing training and validation losses and anything else we might want)
        - data_metadata.pkl (for storing info about how to obtain training and validation data)
    """
    folder = ronn.reduction_method.folder_prefix + NN_FOLDER

    # make folder for training checkpoints
    if not os.path.exists(folder):
        os.mkdir(folder)

    folder = folder + "/" + ronn.name()
    if not os.path.exists(folder):
        os.mkdir(folder)

    # save info for training and validation sets if not already saved
    data_folder = folder + "/data_metadata"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

        torch.save(data.train_data, data_folder + "/train_data.pt")
        torch.save(data.val_data, data_folder + "/val_data.pt")
        torch.save(data.train_data_no_snaps, data_folder + "/train_data_no_snaps.pt")
        torch.save(data.val_data_no_snaps, data_folder + "/val_data_no_snaps.pt")

        # save the non-Tensor attributes
        data_metadata = {
            "loss_type": ronn.loss_type,
            "validation_proportion": data.validation_proportion,
            "initialized": data.initialized,
            "num_without_snapshots": data.num_without_snapshots
        }
        with open(data_folder + "/basic_attributes.pkl", 'wb') as f:
            pickle.dump(data_metadata, f)

    # save the model
    torch.save({
        'epoch': epoch,
        'model_state_dict': ronn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, folder + f"/checkpoint.pt")


    current_train_loss = train_loss_fn.value
    if val_loss_fn is not None:
        current_val_loss = val_loss_fn.value

    metadata = {
        'epoch': epoch,
        'epochs': epochs,
        'train_loss': current_train_loss,
        'train_losses': train_losses,
        'train_loss_type': train_loss_fn.name()
    }

    if val_loss_fn is not None:
        metadata['validation_loss'] = current_val_loss
        metadata['validation_losses'] = validation_losses
        metadata['validation_loss_type'] = val_loss_fn.name()

    if os.path.exists(folder + "/metadata.pkl"):
        os.remove(folder + "/metadata.pkl")

    with open(folder + "/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)



def load_state(ronn, data, trainer, optimizer):
    folder = ronn.reduction_method.folder_prefix + NN_FOLDER + "/" + ronn.name()

    # load the data attributes
    data_folder = folder + "/data_metadata"
    data.train_data = torch.load(data_folder + "/train_data.pt")
    data.val_data = torch.load(data_folder + "/val_data.pt")
    data.train_data_no_snaps = torch.load(data_folder + "/train_data_no_snaps.pt")
    data.val_data_no_snaps = torch.load(data_folder + "/val_data_no_snaps.pt")
    with open(data_folder + "/basic_attributes.pkl", 'rb') as f:
        data_metadata = pickle.load(f)
    data.validation_proportion = data_metadata["validation_proportion"]
    data.initialized = data_metadata["initialized"]
    data.num_without_snapshots = data_metadata["num_without_snapshots"]
    ronn.loss_type = data_metadata["loss_type"]
    with open(folder + "/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    epoch = metadata["epoch"]

    # load the model and optimizer parameters
    checkpoint = torch.load(folder + f"/checkpoint.pt")
    ronn.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs, train_losses, validation_losses = read_losses(ronn)
    trainer.epochs = epochs
    trainer.train_losses = train_losses
    trainer.validation_losses = validation_losses
    if validation_losses is not None:
        if type(validation_losses[-1]) is not dict:
            trainer.best_validation_loss = validation_losses[-1]
        else:
            trainer.best_validation_loss = validation_losses[-1]["loss"]

    ronn.train()
    return epoch

def read_losses(ronn):
    folder = ronn.reduction_method.folder_prefix + NN_FOLDER + "/" + ronn.name()

    train_losses, validation_losses = None, None

    with open(folder + "/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
        train_losses = metadata["train_losses"]
        epochs = metadata["epochs"]
        if "validation_losses" in metadata:
            validation_losses = metadata["validation_losses"]

    return epochs, train_losses, validation_losses

def initialize_parameters(ronn, data, trainer, optimizer):
    loaded_previous_parameters = False

    if os.path.exists(ronn.reduction_method.folder_prefix + NN_FOLDER + "/" + ronn.name()):
        starting_epoch = load_state(ronn, data, trainer, optimizer)
        loaded_previous_parameters = True
    else:
        _ = data.train_validation_split()
        starting_epoch = 0

    return loaded_previous_parameters, starting_epoch
