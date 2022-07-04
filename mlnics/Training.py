import torch
import numpy as np
import matplotlib.pyplot as plt
from mlnics.Normalization import IdentityNormalization
from mlnics.IO import save_state, read_losses_np

NN_FOLDER = "/nn_results"

class RONNTrainer:
    def __init__(self, ronn, data, loss_fn, optimizer=torch.optim.Adam,
                 input_normalization=None, num_epochs=10000, lr=1e-3,
                 print_every=100, starting_epoch=0, use_validation=True):

        ronn.loss_type = loss_fn.name()

        # default initialization for input_normalization
        if input_normalization is None:
            input_normalization = IdentityNormalization()

        self.ronn = ronn
        self.data = data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.input_normalization = input_normalization
        self.num_epochs = num_epochs
        self.lr = lr
        self.print_every = print_every
        self.use_validation = use_validation

        self.best_validation_loss = None
        self.train_losses = []
        self.validation_losses = []
        self.epochs = []

    def train(self):
        """
        Function for getting training and validation sets to pass to
        _normalize_and_train which then performs the training.
        """
        raise NotImplementedError("Cannot train with base class RONNTrainer")

    def _normalize_and_train(self, train, validation, *args):
        """
        If input_normalization has not yet been fit, then this function fits it to the training data.
        """

        starting_epoch = 0 if len(self.epochs) == 0 else self.epochs[-1]+1

        if len(args) > 0:
            assert len(args) == 4
            train_snap, validation_snap, train_no_snap, validation_no_snap = args
            num_train_snaps = train_snap.shape[0]
        else:
            num_train_snaps = train.shape[0]

        train_normalized = self.input_normalization(train) # also initializes normalization
        if validation is not None:
            validation_normalized = self.input_normalization(validation)
            num_validation_snaps = validation.shape[0] if len(args) == 0 else validation_snap.shape[0]
        else:
            validation_normalized = None

        if not self.loss_fn.operators_initialized:
            if len(args) == 0:
                self.loss_fn.set_mu(train)
            else:
                self.loss_fn.set_mu(train_snap, train_no_snap)
            self.loss_fn.slice_snapshots(num_validation_snaps, num_validation_snaps+num_train_snaps)

        # initialize the same loss as loss_fn for validation
        if validation is not None and self.use_validation:
            if len(args) == 0:
                loss_fn_validation = self.loss_fn.reinitialize(validation)
            else:
                loss_fn_validation = self.loss_fn.reinitialize(validation_snap, validation_no_snap)
            loss_fn_validation.slice_snapshots(0, num_validation_snaps)
        else:
            loss_fn_validation = None

        ############################# Train the model #############################

        #best_validation_loss = None
        new_best = False
        #train_losses = []
        #validation_losses = None if validation is None else []
        #epochs = []

        self.optimizer.zero_grad()
        for e in range(starting_epoch, starting_epoch + self.num_epochs):
            coeff_pred = self.ronn(train_normalized)
            loss = self.loss_fn(coeff_pred[:num_train_snaps], coeff_pred[num_train_snaps:], normalized_mu=train_normalized)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if e % self.print_every == 0:
                self.ronn.eval()
                if validation is not None and self.use_validation:
                    pred = self.ronn(validation_normalized)
                    validation_loss = loss_fn_validation(pred[:num_validation_snaps], pred[num_validation_snaps:])
                    if self.best_validation_loss is None or validation_loss.item() <= self.best_validation_loss:
                        self.best_validation_loss = validation_loss.item()
                        new_best = True
                    else:
                        new_best = False

                    self.validation_losses.append(loss_fn_validation.value)
                    print(e, loss.item(), f"\tLoss(validation) = {validation_loss.item()}")
                else:
                    print(e, loss.item())

                self.train_losses.append(self.loss_fn.value)
                self.epochs.append(e)

                if self.best_validation_loss is None or new_best:
                    save_state(e, self.ronn, self.data, self.optimizer, self.loss_fn,
                               loss_fn_validation, self.epochs, self.train_losses, self.validation_losses)
                self.ronn.train()

        self.optimizer.zero_grad()

        return loss_fn_validation


class PDNNTrainer(RONNTrainer):
    def __init__(self, ronn, data, loss_fn, optimizer=torch.optim.Adam,
                 input_normalization=None, num_epochs=10000, lr=1e-3,
                 print_every=100, starting_epoch=0, use_validation=True):

        super(PDNNTrainer, self).__init__(
            ronn, data, loss_fn, optimizer=optimizer,
            input_normalization=input_normalization, num_epochs=num_epochs, lr=lr,
            print_every=print_every, starting_epoch=starting_epoch, use_validation=use_validation
        )

    def train(self):
        """
        Function for getting training and validation sets to pass to
        _normalize_and_train which then performs the training.
        """
        train, validation, _, _ = self.data.train_validation_split()
        return self._normalize_and_train(train, validation)

class PINNTrainer(RONNTrainer):
    def __init__(self, ronn, data, loss_fn, optimizer=torch.optim.Adam,
                 input_normalization=None, num_epochs=10000, lr=1e-3,
                 print_every=100, starting_epoch=0, use_validation=True):

        super(PINNTrainer, self).__init__(
            ronn, data, loss_fn, optimizer=optimizer,
            input_normalization=input_normalization, num_epochs=num_epochs, lr=lr,
            print_every=print_every, starting_epoch=starting_epoch, use_validation=use_validation
        )

    def train(self):
        """
        Function for getting training and validation sets to pass to
        _normalize_and_train which then performs the training.
        """
        _, _, train, validation = self.data.train_validation_split()
        return self._normalize_and_train(train, validation)



class PRNNTrainer(RONNTrainer):
    def __init__(self, ronn, data, loss_fn, optimizer=torch.optim.Adam,
                 input_normalization=None, num_epochs=10000, lr=1e-3,
                 print_every=100, starting_epoch=0, use_validation=True):

        super(PRNNTrainer, self).__init__(
            ronn, data, loss_fn, optimizer=optimizer,
            input_normalization=input_normalization, num_epochs=num_epochs, lr=lr,
            print_every=print_every, starting_epoch=starting_epoch, use_validation=use_validation
        )

    def train(self):
        """
        Function for getting training and validation sets to pass to
        _normalize_and_train which then performs the training.
        """
        train, validation, train_no_snap, val_no_snap = self.data.train_validation_split()
        train_cat = torch.cat([train, train_no_snap], dim=0)
        val_cat = torch.cat([validation, val_no_snap], dim=0)
        return self._normalize_and_train(train_cat, val_cat, train, validation, train_no_snap, val_no_snap)


def plot_loss(trainer, ronn, separate=False):
    epochs, train_losses, validation_losses = trainer.epochs, trainer.train_losses, trainer.validation_losses
    assert len(epochs) > 0

    # make epochs, train_losses, validation_losses the right format
    epochs = np.array(epochs)

    if type(train_losses[0]) is not dict:
        train_losses_ = []
        for i, value in enumerate(train_losses):
            train_losses_.append(value.item())
        train_losses = np.array(train_losses_)

        if validation_losses is not None:
            validation_losses_ = []
            for i, value in enumerate(validation_losses):
                validation_losses_.append(value.item())
            validation_losses = np.array(validation_losses_)
    else:
        train_losses_ = []
        for i, value in enumerate(train_losses):
            train_losses_.append(dict())
            for key in value:
                train_losses_[-1][key] = value[key].item()
        train_dict = dict()
        for key in train_losses[0]:
            train_dict[key] = np.array(list(map(lambda d: d[key], train_losses_)))
        train_losses = train_dict


        if validation_losses is not None:
            validation_losses_ = []
            for i, value in enumerate(validation_losses):
                validation_losses_.append(dict())
                for key in value:
                    validation_losses_[-1][key] = value[key].item()
            validation_dict = dict()
            for key in validation_losses[0]:
                validation_dict[key] = np.array(list(map(lambda d: d[key], validation_losses_)))
            validation_losses = validation_dict

    if not separate:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if type(train_losses) is not dict:
            ax.semilogy(epochs, train_losses, label="Train Loss")
            if validation_losses is not None:
                ax.plot(epochs, validation_losses, label="Validation Loss")
                ax.legend()
        else:
            for key in train_losses:
                ax.semilogy(epochs, train_losses[key], label=f"Train Loss ({key})")
                if validation_losses is not None:
                    ax.semilogy(epochs, validation_losses[key], label=f"Validation Loss ({key})")
                    ax.legend()

        ax.set_title(ronn.name())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
    else:
        if type(train_losses) is not dict:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.semilogy(epochs, train_losses, label="Train Loss")

            if validation_losses is not None:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.semilogy(epochs, validation_losses, label="Validation Loss")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            for key in train_losses:
                ax.semilogy(epochs, train_losses[key], label=f"Train Loss ({key})")

                if validation_losses is not None:
                    ax.semilogy(epochs, validation_losses[key], label=f"Validation Loss ({key})")

        ax.set_title(ronn.name())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    folder = ronn.reduction_method.folder_prefix + NN_FOLDER + "/" + ronn.name()
    fig.savefig(folder + "/loss.png")

    return fig, ax
