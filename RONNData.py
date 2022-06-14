import numpy as np
import torch


class RONNDataLoader:
    def __init__(self, ronn):
        self.ronn = ronn

        self.train_idx = None
        self.train_t0_idx = None
        self.train_data = None

        self.val_idx = None
        self.val_t0_idx = None
        self.val_data = None

        self.validation_proportion = 0.0
        self.initialized = False

    def train_validation_split(self, validation_proportion=0.2):
        if self.initialized:
            assert np.isclose(validation_proportion, self.validation_proportion)
            return (self.train_data, self.val_data)

        self.initialized = True
        self.validation_proportion = validation_proportion

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

            self.val_t0_idx = ronn.num_times * val_idx
            self.train_t0_idx = ronn.num_times * train_idx

            self.val_idx = []
            for idx in self.val_t0_idx:
                for i in range(ronn.num_times):
                    self.val_idx.append(idx+i)
            self.val_idx = torch.tensor(self.val_idx)

            self.train_idx = []
            for idx in self.train_t0_idx:
                for i in range(ronn.num_times):
                    self.train_idx.append(idx+i)
            self.train_idx = torch.tensor(self.train_idx)


            return (self.train_data, self.val_data)
        elif num_validation == 0:
            self.train_data = ronn.augment_parameters_with_time(mu)
            return (self.train_data, None)
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

    def get_validation_initial_time_index(self):
        """
        Returns the indices of the snapshot matrix obtained from self.ronn which
        correspond to the initial times in the validation set.
        """
        ronn = self.ronn

        assert self.initialized
        assert self.val_idx is not None

        if not ronn.time_dependent:
            return None

        T0_idx = torch.zeros(ronn.num_snapshots, dtype=torch.bool)
        T0_idx[torch.arange(0, ronn.num_snapshots, ronn.num_times)] = True
        return T0_idx[self.val_t0_idx].nonzero().view(-1)

    def get_train_initial_time_index(self):
        """
        Returns the indices of the snapshot matrix obtained from self.ronn which
        correspond to the initial times in the training set.
        """
        ronn = self.ronn

        assert self.initialized

        if not ronn.time_dependent:
            return None

        if self.train_t0_idx is None:
            return torch.arange(0, ronn.num_snapshots, ronn.num_times)
        else:
            T0_idx = torch.zeros(ronn.num_snapshots, dtype=torch.bool)
            T0_idx[torch.arange(0, ronn.num_snapshots, ronn.num_times)] = True
            return T0_idx[self.train_t0_idx].nonzero().view(-1)


    def save(self):
        raise NotImplementedError()
