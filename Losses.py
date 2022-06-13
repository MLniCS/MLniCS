import torch
from mlnics.Normalization import IdentityNormalization

class RONN_Loss_Base:
    def __init__(self, ronn):
        self.ronn = ronn
        self.operators_initialized = False

    def __call__(self, pred, **kwargs):
        raise NotImplementedError("Calling abstract method of class RONN_Loss")

class PINN_Loss(RONN_Loss_Base):
    """
    PINN_Loss

    ronn: object of type RONN

    RETURNS: loss function loss_fn(parameters, reduced order coefficients)
    """
    def __init__(self, ronn):
        super(PINN_Loss, self).__init__(ronn)
        self.operators = None
        self.proj_snapshots = None
        self.T0_idx = None
        # if time dependent, we need the neural net to compute time derivative
        self.time_dependent = ronn.time_dependent

    def _compute_operators(self, mu=None):
        print("Getting operator matrices...")
        self.operators = self.ronn.get_operator_matrices()

        if self.ronn.time_dependent:
            print("Getting snapshots...")
            self.proj_snapshots = self.ronn.get_projected_snapshots()
            self.T0_idx = torch.arange(0, self.ronn.num_snapshots, self.ronn.num_times)

    def _batch_jacobian(self, f, x):
        f_sum = lambda x: torch.sum(f(x), axis=0)
        return torch.autograd.functional.jacobian(f_sum, x, create_graph=True)

    def __call__(self, pred, **kwargs):
        if not self.operators_initialized:
            self._compute_operators()
            self.operators_initialized = True

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

            initial_condition_loss = torch.mean((pred[self.T0_idx] - self.proj_snapshots.T[self.T0_idx])**2)
        else:
            initial_condition_loss = 0

        loss1 = torch.mean(torch.sum(res1**2, dim=1)) if type(res1) is not float else res1
        loss2 = torch.mean(torch.sum(res2**2, dim=1)) if type(res2) is not float else res2

        return loss1 + loss2 + initial_condition_loss


class PDNN_Loss(RONN_Loss_Base):
    def __init__(self, ronn, normalization=None):
        super(PDNN_Loss, self).__init__(ronn)
        self.normalization = normalization
        if self.normalization is None:
            self.normalization = IdentityNormalization()

        self.proj_snapshots = None

    def _compute_operators(self):
        if self.normalization is None:
            self.normalization = IdentityNormalization()
        self.proj_snapshots = self.normalization(self.ronn.get_projected_snapshots())

    def __call__(self, pred, **kwargs):
        if not self.operators_initialized:
            self._compute_operators()
            self.operators_initialized = True

        return torch.mean((pred.T - self.proj_snapshots)**2)


class PRNN_Loss(RONN_Loss_Base):
    def __init__(self, ronn, normalization=None, omega=1.):
        super(PRNN_Loss, self).__init__(ronn)
        self.omega = omega
        self.pinn_loss = PINN_Loss(ronn)
        self.pdnn_loss = PDNN_Loss(ronn, normalization)

    def __call__(self, pred, **kwargs):
        return self.pinn_loss(pred, **kwargs) + self.omega * self.pdnn_loss(pred, **kwargs)
