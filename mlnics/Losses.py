import torch
from mlnics.Normalization import IdentityNormalization

class RONN_Loss_Base:
    def __init__(self, ronn, mu=None, snapshot_idx=None):
        self.ronn = ronn
        self.operators_initialized = False
        self.mu = mu
        self.snapshot_idx = snapshot_idx
        self.value = None

    def name(self):
        return "RONN_Base"

    def set_mu(self, mu):
        pass

    def set_snapshot_index(self, idx):
        pass

    def __call__(self, pred, **kwargs):
        raise NotImplementedError("Calling abstract method of class RONN_Loss")

class PINN_Loss(RONN_Loss_Base):
    """
    PINN_Loss

    ronn: object of type RONN

    RETURNS: loss function loss_fn(parameters, reduced order coefficients)
    """
    def __init__(self, ronn, normalization=None, mu=None):
        super(PINN_Loss, self).__init__(ronn, mu)
        self.operators = None
        self.proj_snapshots = None
        self.T0_idx = None
        self.normalization = normalization
        if self.normalization is None:
            self.normalization = IdentityNormalization()

        # if time dependent, we need the neural net to compute time derivative
        self.time_dependent = ronn.time_dependent

    def name(self):
        return "PINN"

    def _compute_operators(self):
        self.operators_initialized = True

        self.operators = self.ronn.get_operator_matrices(self.mu)
        #self.proj_snapshots = self.ronn.get_projected_snapshots()
        #if self.snapshot_idx is not None:
        #    self.proj_snapshots = self.proj_snapshots[:, self.snapshot_idx]
        if self.time_dependent:
            self.T0_snapshots = torch.zeros((self.mu.shape[1], self.mu.shape[0]))
            final_time = self.ronn.reduced_problem.T # store old final time to reset later
            self.ronn.reduced_problem.set_final_time(reduced_problem.t0)
            for i, mu in enumerate(self.mu):
                self.ronn.reduced_problem.set_mu(tuple(np.array(mu)))
                solution = torch.Tensor(reduced_problem.solve()[0].vector()).view(-1, 1)
                self.T0_snapshots[:, i] = solution
            self.ronn.reduced_problem.set_final_time(final_time)

            if not self.normalization.initialized:
                self.normalization(self.proj_snapshots) # fits output normalization to snapshots

            self.T0_idx = torch.arange(0, self.proj_snapshots.shape[1], self.ronn.num_times)

        if not self.normalization.initialized:
            # then it's not a time dependent problem since
            # we did this above for time dependent problems
            self.normalization(self.ronn.get_projected_snapshots())

    def set_mu(self, mu):
        self.mu = mu
        self.operators_initialized = False

    def set_snapshot_index(self, idx):
        self.snapshot_idx = idx
        self.operators_initialized = False

    def _batch_jacobian(self, f, x):
        f_sum = lambda x: torch.sum(self.normalization(f(x).T, normalize=False).T, axis=0)
        return torch.autograd.functional.jacobian(f_sum, x, create_graph=True)

    def __call__(self, pred, **kwargs):
        if not self.operators_initialized:
            self._compute_operators()

        pred = self.normalization(pred.T, normalize=False).T

        ##### 1st equation in system #####
        res1 = 0.0

        # these two could be combined when both not None
        if 'f' in self.operators:
            res1 -= self.operators['f']
        if 'c' in self.operators:
            res1 += self.operators['c']
        # these next two could be combined when they're both not None
        if 'a' in self.operators:
            res1 += torch.matmul(self.operators['a'], pred[:, :, None].double())
        if 'bt' in self.operators:
            res1 += torch.matmul(self.operators['bt'], pred[:, :, None].double())

        ##### 2nd equation in system #####

        res2 = 0.0

        if 'g' in self.operators:
            res2 -= self.operators['g']
        if 'b' in self.operators:
            res2 += torch.matmul(self.operators['b'], pred[:, :, None].double())

        # if time dependent, we include the time derivative in the loss
        # and we include the initial condition in the loss
        if self.time_dependent:
            assert "normalized_mu" in kwargs
            normalized_mu = kwargs["normalized_mu"]
            # perhaps there's a more efficient way to do this?
            # get derivative of neural net output with respect to time

            # this will contain all derivatives of the output with respect to time
            # with shape number of training points x reduced order dimension x 1
            jacobian = torch.permute(self._batch_jacobian(self.ronn, normalized_mu), (1, 0, 2))[:, :, [0]]

            if 'm' in self.operators:
                res1 += torch.matmul(self.operators['m'], jacobian.double())

            #initial_condition_loss = torch.mean((pred[self.T0_idx] - self.proj_snapshots.T[self.T0_idx])**2)
            initial_condition_loss = torch.mean((pred[self.T0_idx] - self.proj_snapshots.T)**2)
        else:
            initial_condition_loss = 0

        loss1 = torch.mean(torch.sum(res1**2, dim=1)) if type(res1) is not float else res1
        loss2 = torch.mean(torch.sum(res2**2, dim=1)) if type(res2) is not float else res2

        self.value = loss1 + loss2 + initial_condition_loss

        return self.value

    def reinitialize(self, mu):
        normalization = self.normalization
        return PINN_Loss(self.ronn, normalization, mu)


class PDNN_Loss(RONN_Loss_Base):
    def __init__(self, ronn, normalization=None, mu=None, snapshot_idx=None):
        super(PDNN_Loss, self).__init__(ronn, mu, snapshot_idx)
        self.normalization = normalization
        if self.normalization is None:
            self.normalization = IdentityNormalization()

        self.proj_snapshots = None

    def name(self):
        return "PDNN"

    def _compute_operators(self):
        self.operators_initialized = True

        if self.normalization is None:
            self.normalization = IdentityNormalization()

        self.proj_snapshots = self.normalization(self.ronn.get_projected_snapshots())
        if self.snapshot_idx is not None:
            self.proj_snapshots = self.proj_snapshots[:, self.snapshot_idx]

    def set_snapshot_index(self, idx):
        self.snapshot_idx = idx

    def __call__(self, pred, **kwargs):
        if not self.operators_initialized:
            self._compute_operators()

        self.value = torch.mean((pred.T - self.proj_snapshots)**2)

        return self.value

    def reinitialize(self, mu, snapshot_idx):
        normalization = self.normalization
        return PDNN_Loss(self.ronn, normalization, mu, snapshot_idx)


class PRNN_Loss(RONN_Loss_Base):
    def __init__(self, ronn, normalization=None, omega=1., mu=None, snapshot_idx=None):
        super(PRNN_Loss, self).__init__(ronn, mu, snapshot_idx)
        self.omega = omega
        self.pinn_loss = PINN_Loss(ronn, normalization, mu)
        self.pdnn_loss = PDNN_Loss(ronn, normalization, mu, snapshot_idx)
        self.value = dict()
        self.value["pinn_loss"] = None
        self.value["pdnn_loss"] = None
        self.value["loss"] = None

    def name(self):
        return f"PRNN_{self.omega}"

    def set_mu(self, pdnn_mu, pinn_mu):
        self.pinn_loss.set_mu(pinn_mu)
        self.pdnn_loss.set_mu(pdnn_mu)

    def set_snapshot_index(self, idx):
        self.pdnn_loss.set_snapshot_index(idx)

    def __call__(self, pdnn_pred, pinn_pred, **kwargs):
        self.value["pdnn_loss"] = self.pdnn_loss(pdnn_pred, **kwargs)
        self.value["pinn_loss"] = self.pinn_loss(pinn_pred, **kwargs)
        self.value["loss"] = self.value["pinn_loss"] + self.omega * self.value["pdnn_loss"]

        return self.value["loss"]

    def reinitialize(self, pdnn_mu, pinn_mu, snapshot_idx):
        normalization = self.pdnn_loss.normalization
        omega = self.omega
        loss = PRNN_Loss(self.ronn, normalization, omega)
        loss.set_mu(pdnn_mu, pinn_mu)
        loss.set_snapshot_index(snapshot_idx)
        return loss
