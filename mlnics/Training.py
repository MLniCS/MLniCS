import torch
import matplotlib.pyplot as plt
from mlnics.Normalization import IdentityNormalization
from mlnics.IO import save_state, read_losses



def normalize_and_train_pdnn(ronn, data, loss_fn, optimizer=torch.optim.Adam, input_normalization=None,
                             epochs=10000, lr=1e-3, print_every=100, starting_epoch=0,
                             folder='./model_checkpoints/', use_validation=True):
    """
    If input_normalization has not yet been fit, then this function fits it to the training data.
    """
    assert data.initialized

    ronn.name = loss_fn.name()

    # default initialization for input_normalization
    if input_normalization is None:
        input_normalization = IdentityNormalization()

    train, validation, _, _ = data.train_validation_split()

    train_normalized = input_normalization(train) # also initializes normalization
    if validation is not None:
        validation_normalized = input_normalization(validation)
    else:
        validation_normalized = None

    val_snapshot_idx = data.get_validation_snapshot_index()
    train_snapshot_idx = data.get_train_snapshot_index()

    if not loss_fn.operators_initialized:
        loss_fn.set_snapshot_index(train_snapshot_idx)
        loss_fn.set_mu(train)

    # initialize the same loss as loss_fn for validation
    if validation is not None and use_validation:
        loss_fn_validation = loss_fn.reinitialize(validation, val_snapshot_idx)
    else:
        loss_fn_validation = None

    ############################# Train the model #############################

    optimizer.zero_grad()
    for e in range(starting_epoch, starting_epoch+epochs):
        coeff_pred = ronn(train_normalized)
        loss = loss_fn(coeff_pred, normalized_mu=train_normalized)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % print_every == 0:
            ronn.eval()
            if validation is not None and use_validation:
                pred = ronn(validation_normalized)
                validation_loss = loss_fn_validation(pred)
                print(e, loss.item(), f"\tLoss(validation) = {validation_loss.item()}")
            else:
                validation_loss = None
                print(e, loss.item())

            save_state(e, ronn, data, optimizer, loss_fn, loss_fn_validation, suffix=loss_fn.name())
            ronn.train()

    optimizer.zero_grad()

    return loss_fn_validation

def normalize_and_train_pinn(ronn, data, loss_fn, optimizer=torch.optim.Adam, input_normalization=None,
                             epochs=10000, lr=1e-3, print_every=100, starting_epoch=0,
                             folder='./model_checkpoints/', use_validation=True):
    """
    If input_normalization has not yet been fit, then this function fits it to the training data.
    """
    assert data.initialized

    ronn.name = loss_fn.name()

    # default initialization for input_normalization
    if input_normalization is None:
        input_normalization = IdentityNormalization()

    _, _, train, validation = data.train_validation_split()

    train_normalized = input_normalization(train) # also initializes normalization
    if validation is not None:
        validation_normalized = input_normalization(validation)
    else:
        validation_normalized = None

    if not loss_fn.operators_initialized:
        loss_fn.set_mu(train)

    # initialize the same loss as loss_fn for validation
    if validation is not None and use_validation:
        loss_fn_validation = loss_fn.reinitialize(validation)
    else:
        loss_fn_validation = None

    ############################# Train the model #############################

    optimizer.zero_grad()
    for e in range(starting_epoch, starting_epoch+epochs):
        coeff_pred = ronn(train_normalized)
        loss = loss_fn(coeff_pred, normalized_mu=train_normalized)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % print_every == 0:
            ronn.eval()
            if validation is not None and use_validation:
                pred = ronn(validation_normalized)
                validation_loss = loss_fn_validation(pred, normalized_mu=validation_normalized)
                print(e, loss.item(), f"\tLoss(validation) = {validation_loss.item()}")
            else:
                validation_loss = None
                print(e, loss.item())

            save_state(e, ronn, data, optimizer, loss_fn, loss_fn_validation, suffix=loss_fn.name())
            ronn.train()

    optimizer.zero_grad()

    return loss_fn_validation

def normalize_and_train_prnn(ronn, data, loss_fn, optimizer=torch.optim.Adam, input_normalization=None,
                             epochs=10000, lr=1e-3, print_every=100, starting_epoch=0,
                             folder='./model_checkpoints/', use_validation=True):
    """
    If input_normalization has not yet been fit, then this function fits it to the training data.
    """
    assert data.initialized

    ronn.name = loss_fn.name()

    # default initialization for input_normalization
    if input_normalization is None:
        input_normalization = IdentityNormalization()

    train, validation, train_no_snap, val_no_snap = data.train_validation_split()

    train_cat = torch.cat([train, train_no_snap], dim=0)
    val_cat = torch.cat([validation, val_no_snap], dim=0)

    train_normalized = input_normalization(train_cat) # also initializes normalization
    if validation is not None:
        validation_normalized = input_normalization(val_cat)
    else:
        validation_normalized = None

    val_snapshot_idx = data.get_validation_snapshot_index()
    train_snapshot_idx = data.get_train_snapshot_index()

    if not loss_fn.operators_initialized:
        loss_fn.set_snapshot_index(train_snapshot_idx)
        loss_fn.set_mu(train, train_no_snap)

    # initialize the same loss as loss_fn for validation
    if validation is not None and use_validation:
        loss_fn_validation = loss_fn.reinitialize(validation, val_no_snap, val_snapshot_idx)
    else:
        loss_fn_validation = None

    ############################# Train the model #############################

    optimizer.zero_grad()
    for e in range(starting_epoch, starting_epoch+epochs):
        coeff_pred = ronn(train_normalized)
        loss = loss_fn(coeff_pred[:train.shape[0]], coeff_pred[train.shape[0]:], normalized_mu=train_normalized)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % print_every == 0:
            ronn.eval()
            if validation is not None and use_validation:
                pred = ronn(validation_normalized)
                validation_loss = loss_fn_validation(pred[:validation.shape[0]], pred[validation.shape[0]:])
                print(e, loss.item(), f"\tLoss(validation) = {validation_loss.item()}")
            else:
                validation_loss = None
                print(e, loss.item())

            save_state(e, ronn, data, optimizer, loss_fn, loss_fn_validation, suffix=loss_fn.name())
            ronn.train()

    optimizer.zero_grad()

    return loss_fn_validation


def plot_loss(ronn, separate=False):
    suffix = ronn.name
    epochs, losses, val_losses = read_losses(ronn, suffix=suffix)

    if not separate:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if type(losses) is not dict:
            ax.semilogy(epochs, losses, label="Train Loss")
            if val_losses is not None:
                ax.plot(epochs, val_losses, label="Validation Loss")
                ax.legend()
        else:
            for key in losses:
                ax.semilogy(epochs, losses[key], label=f"Train Loss ({key})")
                if val_losses is not None:
                    ax.semilogy(epochs, val_losses[key], label=f"Validation Loss ({key})")
                    ax.legend()

        ax.set_title(suffix + " Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        return fig, ax
    else:
        if type(losses) is not dict:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.semilogy(epochs, losses, label="Train Loss")

            if val_losses is not None:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.semilogy(epochs, val_losses, label="Validation Loss")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            for key in losses:
                ax.semilogy(epochs, losses[key], label=f"Train Loss ({key})")

                if val_losses is not None:
                    ax.semilogy(epochs, val_losses[key], label=f"Validation Loss ({key})")

        ax.set_title(suffix + " Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        return fig, ax
