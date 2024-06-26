import numpy as np
import torch
from rbnics.sampling.parameter_space_subset import ParameterSpaceSubset


class RONNDataLoader:
    """
    RONNDataLoader

    The RONNDataLoader class is responsible for loading and dividing the training and validation datasets for the RONN algorithm.

    Parameters

    ronn: RONN object, required
    The RONN object that holds the training data.
    validation_proportion: float, optional (default=0.2)
    The proportion of data to be used as validation data.
    num_without_snapshots: int, optional (default=0)
    The number of samples from the parameter space without corresponding snapshots to be included in the training/validation data.
    sampling: str, optional (default=None)
    The sampling method to be used for generating samples from the parameter space without corresponding snapshots.
    Attributes

    initialized: bool
    A flag indicating whether the training and validation sets have been initialized.
    train_data: torch.tensor
    The training data with corresponding snapshots.
    val_data: torch.tensor
    The validation data with corresponding snapshots.
    train_data_no_snaps: torch.tensor
    The training data without corresponding snapshots.
    val_data_no_snaps: torch.tensor
    The validation data without corresponding snapshots.
    Methods

    train_validation_split():
    Initializes the training and validation datasets, based on the specified parameters.
    get_training_data():
    Returns the training data.
    get_validation_data():
    Returns the validation data.
    get_validation_snapshot_index():
    Returns the validation snapshot index.
    get_train_snapshot_index():
    Returns the training snapshot index.
    get_new_parameters

    The get_new_parameters function returns a torch.tensor of newly generated parameters from the RONN object.

    Parameters

    ronn: RONN object, required
    The RONN object that holds the training data.
    n: int, optional (default=100)
    The number of new parameters to be generated.
    Returns

    torch.tensor
    The newly generated parameters.
    """

    def __init__(self, ronn, validation_proportion=0.2, num_without_snapshots=0, sampling=None):
        self.ronn = ronn

        self.validation_proportion = validation_proportion
        self.initialized = False
        self.train_data = None
        self.val_data = None

        # training/validation sets of parameters without corresponding snapshots
        self.train_data_no_snaps   = None
        self.val_data_no_snaps     = None
        self.num_without_snapshots = num_without_snapshots
        self.sampling              = sampling

    def train_validation_split(self):
        if not self.initialized:
            self.initialized = True

            # first, we initialize training/validation data with corresponding snapshots

            ronn = self.ronn
            mu = ronn.mu # does not contain time as parameter

            # split mu into training and validation sets
            # splitting based on parameters not dependent on time to ensure
            #       that both sets contain initial time if ronn.problem is
            #       time dependent.
            num_validation = int(mu.shape[0]*self.validation_proportion)
            if num_validation > 0 and mu.shape[0] - num_validation > 0:
                self.train_data = ronn.augment_parameters_with_time(mu[num_validation:])
                self.val_data = ronn.augment_parameters_with_time(mu[:num_validation])
                if self.val_data.shape[0] == 0:
                    self.val_data = None
            elif num_validation == 0:
                self.train_data = ronn.augment_parameters_with_time(mu)
            else:
                raise ValueError(f"validation_proportion too large (empty training set).")

            if self.train_data.shape[0] == 0:
                self.train_data = None

            # now we initialize training/validation data without corresponding snapshots
            num_val_without_snaps = int(self.validation_proportion * self.num_without_snapshots)
            num_train_without_snaps = self.num_without_snapshots - num_val_without_snaps

            parameter_space_subset = ParameterSpaceSubset()
            parameter_space_subset.generate(ronn.problem.mu_range, num_val_without_snaps, self.sampling)
            self.val_data_no_snaps = ronn.augment_parameters_with_time(torch.tensor(parameter_space_subset, dtype=torch.float64))
            if self.val_data_no_snaps.shape[0] == 0:
                self.val_data_no_snaps = None

            parameter_space_subset = ParameterSpaceSubset()
            parameter_space_subset.generate(ronn.problem.mu_range, num_train_without_snaps, self.sampling)
            self.train_data_no_snaps = ronn.augment_parameters_with_time(torch.tensor(parameter_space_subset, dtype=torch.float64))
            if self.train_data_no_snaps.shape[0] == 0:
                self.train_data_no_snaps = None

        return (self.train_data, self.val_data, self.train_data_no_snaps, self.val_data_no_snaps)

    def get_training_data(self):
        if not self.initialized:
            ronn = self.ronn
            assert np.isclose(self.validation_proportion, 0.0)
            assert val_data is None
            self.initialized = True
            self.train_data = ronn.augment_parameters_with_time(ronn.mu)
        return self.train_data

    def get_validation_data(self):
        if not self.initialized:
            raise Exception("Training and validation sets not initialized.")
        return self.val_data

    def get_validation_snapshot_index(self):
        return self.val_idx

    def get_train_snapshot_index(self):
        return self.train_idx


def get_new_parameters(ronn, n=100):
    """
    Returns: torch.tensor
    """
    ronn.reduction_method.initialize_testing_set(n)
    mu = torch.tensor(ronn.reduction_method.testing_set)
    return mu
