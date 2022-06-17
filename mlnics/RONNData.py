import numpy as np
import torch


class RONNDataLoader:
    def __init__(self, ronn, validation_proportion=0.2, num_without_snapshots=0, sampling=None):
        self.ronn = ronn

        self.validation_proportion = validation_proportion
        self.initialized = False

        self.train_idx = None
        self.train_data = None

        self.val_idx = None
        self.val_data = None

        # training/validation sets of parameters without corresponding snapshots
        self.train_data_no_snaps = None
        self.val_data_no_snaps = None

    def train_validation_split(self):
        if self.initialized:
            assert np.isclose(validation_proportion, self.validation_proportion)
            return (self.train_data, self.val_data)

        self.initialized = True

        ronn = self.ronn
        mu = ronn.mu # does not contain time as parameter

        # split mu into training and validation sets
        # splitting based on parameters not dependent on time to ensure
        #       that both sets contain initial time if ronn.problem is
        #       time dependent.
        num_validation = int(mu.shape[0]*validation_proportion)
        if num_validation > 0 and mu.shape[0] - num_validation > 0:
            perm = torch.randperm(mu.shape[0])
            val_idx = perm[:num_validation]
            train_idx = perm[num_validation:]
            self.train_data = ronn.augment_parameters_with_time(mu[train_idx])
            self.val_data = ronn.augment_parameters_with_time(mu[val_idx])

            val_t0_idx = ronn.num_times * val_idx
            train_t0_idx = ronn.num_times * train_idx


            self.val_idx = []
            for idx in val_t0_idx:
                for i in range(ronn.num_times):
                    self.val_idx.append(idx+i)
            self.val_idx = torch.tensor(self.val_idx)

            self.train_idx = []
            for idx in train_t0_idx:
                for i in range(ronn.num_times):
                    self.train_idx.append(idx+i)
            self.train_idx = torch.tensor(self.train_idx)

            return (self.train_data, self.val_data)
        elif num_validation == 0:
            self.train_data = ronn.augment_parameters_with_time(mu)
            return (self.train_data, self.val_data)
        else:
            raise ValueError(f"validation_proportion too large (empty training set).")

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

    def save(self):
        raise NotImplementedError()


def get_test(ronn):
    """
    Assumes that testing set has already been initialized.
    Returns: torch.tensor
    """
    mu = torch.tensor(ronn.reduction_method.testing_set)
    return mu
