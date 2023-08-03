import torch
import numpy as np
import matplotlib.pyplot as plt
from mlnics.Normalization import IdentityNormalization
from mlnics.IO import save_state
from tqdm import tqdm

NN_FOLDER = "/nn_results"

class RONNTrainer:
    """
    The RONNTrainer class is a base class for training Reduced Order Nonlinear Networks (RONNs).
    It contains the essential methods for training a RONN with a specified loss function and optimizer,
    and also includes optional components such as a learning rate scheduler and input normalization.
    The class tracks the progress of training by storing the train and validation losses and epochs.

    Parameters:
    ronn (ronn): A RONN model to be trained
    data (tuple): Tuple containing training and validation datasets
    loss_fn (ronn.loss_functions.LossFunction): Loss function for training the RONN
    optimizer (torch.optim.Optimizer): Optimizer for updating the RONN parameters
    lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler for reducing the learning rate
    input_normalization (ronn.normalization.Normalization, optional): Normalization for input data
    num_epochs (int, optional): Number of epochs for training the RONN. Default is 10000.
    lr (float, optional): Learning rate for the optimizer. Default is 1e-3.
    print_every (int, optional): Number of epochs after which the loss is printed. Default is 100.
    starting_epoch (int, optional): Starting epoch number for continuing training from a previous checkpoint. Default is 0.
    use_validation (bool, optional): Flag for using validation data. Default is True.

    Attributes:
    ronn (ronn): A RONN model to be trained
    data (tuple): Tuple containing training and validation datasets
    loss_fn (ronn.loss_functions.LossFunction): Loss function for training the RONN
    optimizer (torch.optim.Optimizer): Optimizer for updating the RONN parameters
    lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler for reducing the learning rate
    input_normalization (ronn.normalization.Normalization, optional): Normalization for input data
    num_epochs (int, optional): Number of epochs for training the RONN
    lr (float, optional): Learning rate for the optimizer
    print_every (int, optional): Number of epochs after which the loss is printed
    use_validation (bool, optional): Flag for using validation data
    best_validation_loss (float, optional): Best validation loss so far
    train_losses (list of float): List of training losses at each epoch
    validation_losses (list of float): List of validation losses at each epoch
    epochs (list of int): List of epoch numbers

    Methods:
    train (): Raises NotImplementedError as the base class does not implement the training method.
    _normalize_and_train (): Performs the normalization of the input data and trains the RONN using the specified loss function and optimizer.
    """

    def __init__(self, ronn, data, loss_fn, optimizer,
                 lr_scheduler=None,
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
        self.lr_scheduler = lr_scheduler
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

    def _normalize_and_train(self, train_snap, validation_snap, train_no_snap, validation_no_snap):
        """
        If input_normalization has not yet been fit, then this function fits it to the training data.
        """

        both_snap_no_snap = False
        if train_snap is None:
            train = train_no_snap
            validation = validation_no_snap
            num_train_snaps = 0
            num_validation_snaps = 0
        elif train_no_snap is None:
            train = train_snap
            validation = validation_snap
            num_train_snaps = train_snap.shape[0]
            if validation_snap is not None:
                num_validation_snaps = validation_snap.shape[0]
            else:
                num_validation_snaps = 0
        else:
            both_snap_no_snap = True
            train = torch.cat([train_snap, train_no_snap], dim=0)
            num_train_snaps = train_snap.shape[0]

            # if the loss function uses both data with snapshots and
            # data without snapshots, then it is necessary to have both
            # types of data for validation
            if validation_snap is not None and validation_no_snap is not None:
                num_validation_snaps = validation_snap.shape[0]
                validation = torch.cat([validation_snap, validation_no_snap], dim=0)
            else:
                num_validation_snaps = 0
                validation = None

        starting_epoch = 0 if len(self.epochs) == 0 else self.epochs[-1]+1

        train_normalized = self.input_normalization(train) # also initializes normalization
        if validation is not None:
            validation_normalized = self.input_normalization(validation)
        else:
            validation_normalized = None

        if not self.loss_fn.operators_initialized:
            if not both_snap_no_snap:
                self.loss_fn.set_mu(train)
            else:
                self.loss_fn.set_mu(train_snap, train_no_snap)
            self.loss_fn.slice_snapshots(num_validation_snaps, num_validation_snaps+num_train_snaps)

        # initialize the same loss as loss_fn for validation
        if validation is not None and self.use_validation:
            if not both_snap_no_snap:
                loss_fn_validation = self.loss_fn.reinitialize(validation)
            else:
                loss_fn_validation = self.loss_fn.reinitialize(validation_snap, validation_no_snap)
            loss_fn_validation.slice_snapshots(0, num_validation_snaps)
        else:
            loss_fn_validation = None

        ############################# Train the model #############################

        new_best = False

        self.optimizer.zero_grad()
        # loop = range(starting_epoch, starting_epoch + self.num_epochs)
        loop = tqdm(range(starting_epoch, starting_epoch + self.num_epochs))
        for e in loop:
            coeff_pred = self.ronn(train_normalized)
            loss = self.loss_fn(prediction_snap=coeff_pred[:num_train_snaps],
                                prediction_no_snap=coeff_pred[num_train_snaps:],
                                input_normalization=self.input_normalization,
                                normalized_mu=train_normalized[num_train_snaps:])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                if type(self.lr_scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.lr_scheduler.step(loss)
                elif type(self.lr_scheduler) is torch.optim.lr_scheduler.ExponentialLR:
                    self.lr_scheduler.step()
                else:
                    raise NotImplementedError(str(type(self.lr_scheduler)) + " not implemented.")

            # loop.set_description(f"Epoch [{e}/{starting_epoch + self.num_epochs}]")
            if e % self.print_every == 0:
                self.ronn.eval()
                if validation is not None and self.use_validation:
                    pred = self.ronn(validation_normalized)
                    validation_loss = loss_fn_validation(prediction_snap=pred[:num_validation_snaps],
                                                         prediction_no_snap=pred[num_validation_snaps:],
                                                         input_normalization=self.input_normalization,
                                                         normalized_mu=validation_normalized[num_validation_snaps:])
                    if self.best_validation_loss is None or validation_loss.item() <= self.best_validation_loss:
                        self.best_validation_loss = validation_loss.item()
                        new_best = True
                    else:
                        new_best = False

                    self.validation_losses.append(loss_fn_validation.value)
                    # print(e, f"\tLoss(training) = {loss.item()}", f"\tLoss(validation) = {validation_loss.item()}")
                    loop.set_postfix({"Loss(training)": loss.item()}, {"Loss(validation)": validation_loss.item()})
                else:
                    # print(e, f"\tLoss(training) = {loss.item()}")
                    loop.set_postfix({"Loss(training)": loss.item()})

                self.train_losses.append(self.loss_fn.value)
                self.epochs.append(e)

                if self.best_validation_loss is None or new_best:
                    save_state(e, self.ronn, self.data, self.optimizer, self.loss_fn,
                               loss_fn_validation, self.epochs, self.train_losses, self.validation_losses)
                self.ronn.train()

        self.optimizer.zero_grad()

        return loss_fn_validation


class PDNNTrainer(RONNTrainer):
    """
    PDNNTrainer

    This class is a child class of the RONNTrainer class. It is used to train a
    RONN (Radial Overlap Neural Network) for prediction of chemical properties using
    the PDNN (Predictive Deep Neural Network) training method.

    The class provides a train method which trains the RONN model using the PDNN method.

    Parameters:

    ronn (RONN): The RONN model to be trained.
    data (Data): The data to be used for training.
    loss_fn (function): The loss function to be used for training.
    optimizer (function): The optimizer function to be used for training.
    lr_scheduler (function, optional): The learning rate scheduler to be used. Defaults to None.
    input_normalization (function, optional): The input normalization function to be used. Defaults to None.
    num_epochs (int, optional): The number of training epochs. Defaults to 10000.
    lr (float, optional): The learning rate. Defaults to 1e-3.
    print_every (int, optional): The number of epochs after which training progress is printed. Defaults to 100.
    starting_epoch (int, optional): The starting epoch number. Defaults to 0.
    use_validation (bool, optional): Whether to use validation set during training. Defaults to True.

    Methods:
    train(): Trains the RONN model using the PDNN method. Returns the training and validation results.
    """

    def __init__(self, ronn, data, loss_fn, optimizer, lr_scheduler=None,
                 input_normalization=None, num_epochs=10000, lr=1e-3,
                 print_every=100, starting_epoch=0, use_validation=True):

        super(PDNNTrainer, self).__init__(
            ronn, data, loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler,
            input_normalization=input_normalization, num_epochs=num_epochs, lr=lr,
            print_every=print_every, starting_epoch=starting_epoch, use_validation=use_validation
        )

    def train(self):
        """
        Function for getting training and validation sets to pass to
        _normalize_and_train which then performs the training.
        """
        train_snap, validation_snap, _, _ = self.data.train_validation_split()
        return self._normalize_and_train(train_snap, validation_snap, None, None)


class PINNTrainer(RONNTrainer):
    """
    PINNTrainer

    This class is a child class of the RONNTrainer class. It is used to train a
    RONN (Radial Overlap Neural Network) for prediction of chemical properties using
    the PINN (Predictive Neural Network) training method.

    The class provides a train method which trains the RONN model using the PINN method.

    Parameters:

    ronn (RONN): The RONN model to be trained.
    data (Data): The data to be used for training.
    loss_fn (function): The loss function to be used for training.
    optimizer (function): The optimizer function to be used for training.
    lr_scheduler (function, optional): The learning rate scheduler to be used. Defaults to None.
    input_normalization (function, optional): The input normalization function to be used. Defaults to None.
    num_epochs (int, optional): The number of training epochs. Defaults to 10000.
    lr (float, optional): The learning rate. Defaults to 1e-3.
    print_every (int, optional): The number of epochs after which training progress is printed. Defaults to 100.
    starting_epoch (int, optional): The starting epoch number. Defaults to 0.
    use_validation (bool, optional): Whether to use validation set during training. Defaults to True.

    Methods:
    train(): Trains the RONN model using the PINN method. Returns the training and validation results.
    """

    def __init__(self, ronn, data, loss_fn, optimizer, lr_scheduler=None,
                 input_normalization=None, num_epochs=10000, lr=1e-3,
                 print_every=100, starting_epoch=0, use_validation=True):

        super(PINNTrainer, self).__init__(
            ronn, data, loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler,
            input_normalization=input_normalization, num_epochs=num_epochs, lr=lr,
            print_every=print_every, starting_epoch=starting_epoch, use_validation=use_validation
        )

    def train(self):
        """
        Function for getting training and validation sets to pass to
        _normalize_and_train which then performs the training.
        """
        _, _, train_no_snap, validation_no_snap = self.data.train_validation_split()
        return self._normalize_and_train(None, None, train_no_snap, validation_no_snap)


class PRNNTrainer(RONNTrainer):
    """
    Class PRNNTrainer

    A class that extends `RONNTrainer` and trains a `PRNN` model.

    Attributes:
        ronn (RONN): The RONN model to be trained.
        data (object): The data to be used for training.
        loss_fn (function): The loss function to be used for training.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        lr_scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler to be used for training. Defaults to None.
        input_normalization (callable, optional): The input normalization function to be used for training. Defaults to None.
        num_epochs (int, optional): The number of epochs for training. Defaults to 10000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
        print_every (int, optional): The interval for printing training loss. Defaults to 100.
        starting_epoch (int, optional): The starting epoch number. Defaults to 0.
        use_validation (bool, optional): Whether to use validation during training. Defaults to True.

    Methods:
        train(): Function that gets training and validation sets to pass to _normalize_and_train which then performs the training.

    Function plot_loss

    Plots the loss curve for a given `PRNNTrainer` object.

    Args:
        trainer (PRNNTrainer): The PRNNTrainer object for which the loss curve is to be plotted.
        ronn (RONN): The RONN model used in the trainer.
        separate (bool, optional): Whether to plot train and validation losses on separate plots. Defaults to False.
    """

    def __init__(self, ronn, data, loss_fn, optimizer, lr_scheduler=None,
                 input_normalization=None, num_epochs=10000, lr=1e-3,
                 print_every=100, starting_epoch=0, use_validation=True):

        super(PRNNTrainer, self).__init__(
            ronn, data, loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler,
            input_normalization=input_normalization, num_epochs=num_epochs, lr=lr,
            print_every=print_every, starting_epoch=starting_epoch, use_validation=use_validation
        )

    def train(self):
        """
        Function for getting training and validation sets to pass to
        _normalize_and_train which then performs the training.
        """
        train_snap, validation_snap, train_no_snap, val_no_snap = self.data.train_validation_split()
        return self._normalize_and_train(train_snap, validation_snap, train_no_snap, val_no_snap)


def plot_loss(trainer, ronn, separate=False):
    epochs, train_losses, validation_losses = trainer.epochs, trainer.train_losses, trainer.validation_losses
    assert len(epochs) > 0

    # make epochs, train_losses, validation_losses the right format
    epochs = np.array(epochs)

    if type(train_losses[0]) is not dict:
        train_losses_ = []
        for i, value in enumerate(train_losses):
            train_losses_.append(value)
        train_losses = np.array(train_losses_)

        if validation_losses is not None:
            validation_losses_ = []
            for i, value in enumerate(validation_losses):
                validation_losses_.append(value)
            validation_losses = np.array(validation_losses_)
    else:
        train_losses_ = []
        for i, value in enumerate(train_losses):
            train_losses_.append(dict())
            for key in value:
                train_losses_[-1][key] = value[key]
        train_dict = dict()
        for key in train_losses[0]:
            train_dict[key] = np.array(list(map(lambda d: d[key], train_losses_)))
        train_losses = train_dict


        if validation_losses is not None:
            validation_losses_ = []
            for i, value in enumerate(validation_losses):
                validation_losses_.append(dict())
                for key in value:
                    validation_losses_[-1][key] = value[key]
            validation_dict = dict()
            if len(validation_losses) > 0:
                for key in validation_losses[0]:
                    validation_dict[key] = np.array(list(map(lambda d: d[key], validation_losses_)))
            validation_losses = validation_dict

    if not separate:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if type(train_losses) is not dict:
            ax.semilogy(epochs, train_losses, linestyle='dashed', label="Train Loss")
            if validation_losses is not None and np.size(validation_losses) > 0:
                ax.plot(epochs, validation_losses, linestyle='solid', label="Validation Loss")
                ax.legend()
        else:
            for key in train_losses:
                ax.semilogy(epochs, train_losses[key], linestyle='dashed', label=f"Train Loss ({key})")
                if validation_losses is not None and len(validation_losses) > 0:
                    ax.semilogy(epochs, validation_losses[key], label=f"Validation Loss ({key})")
                    ax.legend()

        ax.set_title(ronn.name())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
    else:
        if type(train_losses) is not dict:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.semilogy(epochs, train_losses, linestyle='dashed', label="Train Loss")

            if validation_losses is not None and np.size(validation_losses) > 0:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.semilogy(epochs, validation_losses, linestyle='solid', label="Validation Loss")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            for key in train_losses:
                ax.semilogy(epochs, train_losses[key], linestyle='dashed', label=f"Train Loss ({key})")

                if validation_losses is not None and len(validation_losses) > 0:
                    ax.semilogy(epochs, validation_losses[key], linestyle='solid', label=f"Validation Loss ({key})")

        ax.set_title(ronn.name())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    folder = ronn.reduction_method.folder_prefix + NN_FOLDER + "/" + ronn.name()
    fig.savefig(folder + "/loss.png")

    return fig, ax
