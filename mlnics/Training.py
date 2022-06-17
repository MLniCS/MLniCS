import torch
import matplotlib.pyplot as plt
from mlnics.Losses import reinitialize_loss
from mlnics.Normalization import IdentityNormalization
from mlnics.IO import save_state, read_losses



def normalize_and_train(ronn, data, loss_fn, optimizer=torch.optim.Adam, input_normalization=None,
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

    train, validation = data.train_validation_split()
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
        loss_fn_validation = reinitialize_loss(ronn, loss_fn, validation, val_snapshot_idx)
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

def plot_loss(ronn, separate=False):
    suffix = ronn.name
    epochs, losses, val_losses = read_losses(ronn, suffix=suffix)

    if not separate:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if type(losses) is not dict:
            ax.plot(epochs, losses, label="Train Loss")
            if val_losses is not None:
                ax.plot(epochs, val_losses, label="Validation Loss")
                ax.legend()
        else:
            for key in losses:
                ax.plot(epochs, losses[key], label=f"Train Loss ({key})")
                if val_losses is not None:
                    ax.plot(epochs, val_losses[key], label=f"Validation Loss ({key})")
                    ax.legend()

        ax.set_title(suffix + " Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        return fig, ax
    else:
        plot_list = []
        if type(losses) is not dict:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(epochs, losses, label="Train Loss")
            plot_list.append((fig, ax))

            if val_losses is not None:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(epochs, val_losses, label="Validation Loss")
                plot_list.append((fig, ax))
        else:
            for key in losses:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(epochs, losses[key], label=f"Train Loss ({key})")
                plot_list.append((fig, ax))

                if val_losses is not None:
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(epochs, val_losses[key], label=f"Validation Loss ({key})")
                    plot_list.append((fig, ax))

        for (fig, ax) in plot_list:
            ax.set_title(suffix + " Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()

        return plot_list
