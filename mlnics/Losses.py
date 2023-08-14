import torch
import numpy as np
import copy
from mlnics.Normalization import IdentityNormalization

from rbnics.backends.basic.wrapping.delayed_transpose import DelayedTranspose
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends import evaluate



class RONN_Loss_Base:
    """
    RONN_Loss_Base

    This is the base class for all RONN loss functions. It contains common functionality for all RONN losses.

    Parameters

    ronn : object
    An instance of the RONN class that has been pre-trained on a dataset.
    mu : float, optional
    A hyperparameter for the loss function (default is None).

    Attributes

    ronn : object
    An instance of the RONN class that has been pre-trained on a dataset.
    operators_initialized : bool
    A flag indicating whether the RONN operators have been initialized.
    mu : float
    A hyperparameter for the loss function.
    value : float
    The value of the loss function.

    Methods

    name()
    Returns the name of the RONN loss function.
    set_mu(mu)
    Sets the value of the mu hyperparameter.
    slice_snapshots(start, end)
    Slices the RONN snapshots for a given range.
    call(pred, kwargs)
    Returns the value of the loss function for a given prediction.

    Raises

    NotImplementedError
    If the call method is called on an instance of the RONN_Loss_Base class without being implemented by a subclass.
    """

    def __init__(self, ronn, mu=None):
        self.ronn = ronn
        self.operators_initialized = False
        self.mu = mu
        self.value = None

    def name(self):
        return "RONN_Base"

    def set_mu(self, mu):
        pass

    def slice_snapshots(self, start, end):
        pass

    def __call__(self, pred, **kwargs):
        raise NotImplementedError("Calling abstract method of class RONN_Loss")


class PINN_Loss(RONN_Loss_Base):
    """
    Class PINN_Loss

    This class defines the PINN loss function used for training neural networks with the RONN method.

    Attributes:
    ronn (object): an object of type RONN, representing the reduced order model.
    normalization (optional): an object for normalization, by default None.
    beta (float): parameter used in the loss function, by default 1.0.
    mu (optional): the parameters used in the reduced order model, by default None.
    DEIM_func_c (optional): the nonlinearity used in the reduced order model, by default None.
    DEIM_func_f (optional): the nonlinearity used in the reduced order model, by default None.

    Methods:
    init(self, ronn, normalization=None, beta=1., mu=None, DEIM_func_c=None, DEIM_func_f=None): Initializes the attributes.
    name(self): returns the name of the loss function as a string.
    _compute_operators(self): computes the reduced order operator matrices.

    Returns:
    loss_fn (function): the loss function loss_fn(parameters, reduced order coefficients).
    """

    def __init__(self, ronn, normalization=None, beta=1., mu=None, DEIM_func_c=None, DEIM_func_f=None, func_c=None):
        super(PINN_Loss, self).__init__(ronn, mu)
        self.operators = None
        self.proj_snapshots = None
        self.T0_idx = None
        self.normalization = normalization
        if self.normalization is None:
            self.normalization = IdentityNormalization()

        self.beta = beta

        # if time dependent, we need the neural net to compute time derivative
        self.time_dependent = ronn.time_dependent

        # setup for nonlinearity
        self.using_DEIM_c = False
        self.using_c = False
        if DEIM_func_c is not None:
            self.nonlinearity = DEIM_func_c
            self.using_DEIM_c = True
        elif func_c is not None:
            self.nonlinearity = func_c
            self.using_c = True
        else:
            self.nonlinearity = None

        if DEIM_func_f is not None:
            self.nonlinearity_f = DEIM_func_f
            self.using_DEIM_f = True
        else:
            self.nonlinearity_f = None
            self.using_DEIM_f = False

    def name(self):
        return "PINN"

    def _compute_operators(self):
        self.operators_initialized = True
        self.operators = self.ronn.get_reduced_operator_matrices(self.mu)
        if self.using_DEIM_c:
            # make the operator for DEIM
            selected_indices = sorted([idx[0] for idx in self.ronn.problem.DEIM_approximations['c'][0].interpolation_locations.get_dofs_list()])
            U = np.array(self.ronn.problem._assemble_operator_DEIM('c')).T
            P = []
            for idx in selected_indices:
                new_column = np.zeros(U.shape[0])
                new_column[idx] = 1
                P.append(new_column)
            P = np.array(P).T
            PtUinv = np.linalg.inv(P.T @ U)

            self.operators['c'] = torch.tensor(self.ronn.projection @ U @ PtUinv, dtype=torch.float64)
            self.basis_matrix = torch.tensor(self.ronn.projection.T[selected_indices], dtype=torch.float64)

        if self.using_DEIM_f:
            self.using_DEIM_f = True
            # make the operator for DEIM
            num_cols = len(self.ronn.reduced_problem.operator['f'])
            CT = np.zeros((num_cols, self.ronn.ro_dim))
            for j, Cj in enumerate(self.ronn.reduced_problem.operator['f']):
                if type(Cj) is ParametrizedTensorFactory:
                    Cj = np.array(evaluate(Cj))
                elif type(Cj) is DelayedTranspose:
                    Cj = (np.array([v.vector() for v in Cj._args[0]]) @ np.array(evaluate(Cj._args[1])).reshape(-1, 1)).reshape(-1)
                else:
                    Cj = Cj.reshape(-1)[0].content
                CT[j] = Cj

            self.operators['f'] = torch.tensor(CT.T, dtype=torch.float64)
        
        if self.using_c:
            self.operators['c'] = None
            self.basis_matrix = torch.tensor(self.ronn.projection.T, dtype=torch.float64)
            self.basis_matrix2 = self.ronn.get_coefficient_matrix()

        if self.time_dependent:
            self.T0_idx = torch.arange(0, self.mu.shape[0], self.ronn.num_times)

            self.T0_snapshots = torch.zeros((self.ronn.ro_dim, self.T0_idx.shape[0]), dtype=torch.float64)
            
            sotc = copy.deepcopy(self.ronn.reduced_problem._solution_over_time_cache)
            sdotc = copy.deepcopy(self.ronn.reduced_problem._solution_dot_over_time_cache)
            ootc = copy.deepcopy(self.ronn.reduced_problem._output_over_time_cache)
            
            final_time = self.ronn.reduced_problem.T # store old final time to reset later
            self.ronn.reduced_problem.set_final_time(self.ronn.reduced_problem.t0)
            for i, mu in enumerate(self.mu[self.T0_idx]):
                self.ronn.reduced_problem.set_mu(tuple(np.array(mu)[1:]))
                solution = torch.tensor(np.array(self.ronn.reduced_problem.solve()[0].vector()), dtype=torch.float64).view(-1, 1)
                self.T0_snapshots[:, [i]] = solution
            
            self.ronn.reduced_problem._solution_over_time_cache = sotc
            self.ronn.reduced_problem._solution_dot_over_time_cache = sdotc
            self.ronn.reduced_problem._output_over_time_cache = ootc
            self.ronn.reduced_problem.set_final_time(final_time)
            

        if not self.normalization.initialized:
            self.normalization(self.ronn.get_projected_snapshots())

    def set_mu(self, mu):
        self.mu = mu
        self.operators_initialized = False

    def _batch_jacobian(self, f, x, input_normalization):
        f_sum = lambda y: torch.sum(self.normalization(f(input_normalization(y)).T, normalize=False).T, axis=0)
        return torch.autograd.functional.jacobian(f_sum, x, create_graph=True)

    def __call__(self, **kwargs):
        pred = kwargs["prediction_no_snap"]
        if not self.operators_initialized:
            self._compute_operators()

        pred = self.normalization(pred.T, normalize=False).T

        ##### 1st equation in system #####
        res1 = 0.0

        # these two could be combined when both not None
        if 'f' in self.operators:
            if self.using_DEIM_f:
                res1 -= torch.matmul(self.operators['f'],
                            self.nonlinearity_f(
                                kwargs["input_normalization"](kwargs["normalized_mu"], normalize=False)
                            )
                        ).T[:, :, None]
            else:
                res1 -= self.operators['f']
        if 'c' in self.operators:
            if self.using_DEIM_c:
                tmp = torch.matmul(self.operators['c'],
                            self.nonlinearity(
                                torch.matmul(self.basis_matrix, pred.T.double()),
                                kwargs["input_normalization"](kwargs["normalized_mu"], normalize=False)
                            )
                        ).T[:, :, None]
                res1 += tmp
            elif self.using_c:
                tmp = torch.matmul(self.basis_matrix2.T,
                            self.nonlinearity(
                                torch.matmul(self.basis_matrix2, pred.T.double()),
                                kwargs["input_normalization"](kwargs["normalized_mu"], normalize=False)
                            )
                        ).T[:, :, None]
                res1 += tmp
            else:
                raise ValueError("Nonlinearities must be provided by user at initialization of loss.")
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

            input_normalization = kwargs["input_normalization"]
            # perhaps there's a more efficient way to do this?
            # get derivative of neural net output with respect to time

            # this will contain all derivatives of the output with respect to time
            # with shape number of training points x reduced order dimension x 1
            jacobian = torch.permute(self._batch_jacobian(self.ronn, input_normalization(kwargs["normalized_mu"], normalize=False), input_normalization), (1, 0, 2))[:, :, [0]]

            if 'm' in self.operators:
                res1 += torch.matmul(self.operators['m'], jacobian.double())

            initial_condition_loss = torch.mean((pred[self.T0_idx] - self.T0_snapshots.T)**2)
        else:
            initial_condition_loss = 0
        
        loss1 = torch.mean(torch.sum(res1**2, dim=1)) if type(res1) is not float else res1
        loss2 = torch.mean(torch.sum(res2**2, dim=1)) if type(res2) is not float else res2
        
        if self.ronn.problem.dirichlet_bc is not None and not self.ronn.problem.dirichlet_bc_are_homogeneous:
            boundary_condition_loss = torch.mean((pred[:, 0] - 1.)**2)
        else:
            boundary_condition_loss = 0
        
        val = loss1 + loss2 + initial_condition_loss + self.beta*boundary_condition_loss
        self.value = val.item()

        return val

    def reinitialize(self, mu):
        normalization = self.normalization
        beta = self.beta
        nonlinearity = self.nonlinearity
        return PINN_Loss(self.ronn, normalization, beta, mu, nonlinearity)


class PDNN_Loss(RONN_Loss_Base):
    """
    This class extends the RONN_Loss_Base class to implement the PDNN loss.

    Parameters:
    ronn (RONN_Loss_Base): The base RONN loss class.
    normalization (None, optional): The normalization method to use. If None, it will use the IdentityNormalization method. Defaults to None.
    mu (None, optional): The mu parameter used in the RONN loss. Defaults to None.

    Methods:
    name: Returns the string "PDNN".
    slice_snapshots: Slices the snapshots from the start to the end indices.
    concatenate_snapshots: Concatenates the snapshots with the normalization applied.
    _compute_operators: Computes the necessary operators for the PDNN loss.
    call: Computes the mean squared error between the prediction and the projected snapshots.
    reinitialize: Reinitializes the PDNN loss with a new mu value.
    """

    def __init__(self, ronn, normalization=None, mu=None):
        super(PDNN_Loss, self).__init__(ronn, mu)
        self.normalization = normalization
        if self.normalization is None:
            self.normalization = IdentityNormalization()

        self.proj_snapshots = None

    def name(self):
        return "PDNN"

    def slice_snapshots(self, start, end):
        assert not self.operators_initialized
        self._compute_operators()
        self.proj_snapshots = self.proj_snapshots[:, start:end]

    def concatenate_snapshots(self, snapshots):
        assert self.operators_initialized
        self.proj_snapshots = torch.cat([self.proj_snapshots, self.normalization(snapshots)], dim=1)

    def _compute_operators(self):
        self.operators_initialized = True

        if self.normalization is None:
            self.normalization = IdentityNormalization()

        self.proj_snapshots = self.normalization(self.ronn.get_projected_snapshots())

    def __call__(self, **kwargs):
        pred = kwargs["prediction_snap"]
        if not self.operators_initialized:
            self._compute_operators()

        val = torch.mean((pred.T - self.proj_snapshots)**2)
        self.value = val.item()

        return val

    def reinitialize(self, mu):
        normalization = self.normalization
        return PDNN_Loss(self.ronn, normalization, mu)


class PRNN_Loss(RONN_Loss_Base):
    """
    PRNN_Loss class extends the RONN_Loss_Base class and calculates the loss function for a Physics-Regularized Neural Network (PRNN) model.

    Attributes:
    omega (float): A scalar weight for the physics-based loss.
    beta (float): A scalar weight for the PINN loss.
    pinn_loss (PINN_Loss): An instance of the PINN_Loss class to calculate the PINN loss.
    pdnn_loss (PDNN_Loss): An instance of the PDNN_Loss class to calculate the physics-based loss.
    value (dict): A dictionary to store the values of the PINN loss, physics-based loss, and the total loss.

    Methods:
    name (): Returns a string representing the name of the loss function with the value of omega.
    set_mu (pdnn_mu, pinn_mu): Sets the values of mu for both the PINN loss and physics-based loss.
    slice_snapshots (start, end): Slices the snapshots used for the calculation of the physics-based loss.
    call (kwargs): Calculates and returns the total loss.
    reinitialize (pdnn_mu, pinn_mu): Re-initializes the loss function with the given values of mu for the PINN loss and physics-based loss.

    Note:
    The RONN_Loss_Base class and the PINN_Loss and PDNN_Loss classes are required for the PRNN_Loss class to work properly.
    """

    def __init__(self, ronn, normalization=None, omega=1., beta=1., mu=None):
        super(PRNN_Loss, self).__init__(ronn, mu)
        self.omega = omega
        self.beta = beta
        self.pinn_loss = PINN_Loss(ronn, normalization, beta, mu)
        self.pdnn_loss = PDNN_Loss(ronn, normalization, mu)
        self.value = dict()
        self.value["pinn_loss"] = None
        self.value["pdnn_loss"] = None
        self.value["loss"] = None

    def name(self):
        return f"PRNN_{self.omega}"

    def set_mu(self, pdnn_mu, pinn_mu):
        self.pinn_loss.set_mu(pinn_mu)
        self.pdnn_loss.set_mu(pdnn_mu)

    def slice_snapshots(self, start, end):
        self.pdnn_loss.slice_snapshots(start, end)

    def __call__(self, **kwargs):
        self.operators_initialized = True

        pdnn_val = self.pdnn_loss(**kwargs)
        pinn_val = self.pinn_loss(**kwargs)
        val = pinn_val + self.omega * pdnn_val

        self.value = dict()
        self.value["pdnn_loss"] = pdnn_val.item()
        self.value["pinn_loss"] = pinn_val.item()
        self.value["loss"] = val.item()

        return val

    def reinitialize(self, pdnn_mu, pinn_mu):
        normalization = self.pdnn_loss.normalization
        omega = self.omega
        beta = self.beta
        loss = PRNN_Loss(self.ronn, normalization, omega, beta)
        loss.set_mu(pdnn_mu, pinn_mu)
        return loss


""" Other losses """
class Weighted_PDNN_Loss(RONN_Loss_Base):
    """
    This class implements the Weighted PDNN Loss. It is a sub-class of RONN Loss Base and is used to calculate the loss value in a Reduced Order Non-Linear network (RONN).

    The class uses normalization and epsilon to calculate the weighted mean of the mean squared differences between the predicted snapshot and the projected snapshots. The epsilon value determines the rate at which the weights decay over time.

    Attributes:
    normalization (torch.nn.Module): Normalization applied to the projected snapshots.
    proj_snapshots (torch.Tensor): Projected snapshots after normalization.
    epsilon (float): Decay rate of the weights.

    Methods:
    name: Returns the string 'PDNN' as the name of the loss.
    slice_snapshots: Slice the projected snapshots based on start and end indices.
    concatenate_snapshots: Concatenate the snapshots with the existing projected snapshots.
    _compute_operators: Compute the normalization and projected snapshots.
    call: Calculate the weighted mean of mean squared differences as the loss value.
    reinitialize: Reinitialize the Weighted PDNN Loss with a new value for mu.
    """

    def __init__(self, ronn, normalization=None, mu=None, epsilon=100.):
        super(Weighted_PDNN_Loss, self).__init__(ronn, mu)
        self.normalization = normalization
        if self.normalization is None:
            self.normalization = IdentityNormalization()

        self.proj_snapshots = None
        self.epsilon = epsilon

    def name(self):
        return "PDNN"

    def slice_snapshots(self, start, end):
        assert not self.operators_initialized
        self._compute_operators()
        self.proj_snapshots = self.proj_snapshots[:, start:end]

    def concatenate_snapshots(self, snapshots):
        assert self.operators_initialized
        self.proj_snapshots = torch.cat([self.proj_snapshots, self.normalization(snapshots)], dim=1)

    def _compute_operators(self):
        self.operators_initialized = True

        if self.normalization is None:
            self.normalization = IdentityNormalization()

        self.proj_snapshots = self.normalization(self.ronn.get_projected_snapshots())

    def __call__(self, **kwargs):
        pred = kwargs["prediction_snap"]
        if not self.operators_initialized:
            self._compute_operators()

        separate_losses = torch.mean((pred.T - self.proj_snapshots)**2, dim=0)
        weights = torch.exp(-self.epsilon * torch.cumsum(separate_losses, dim=0))
        separate_losses[1:] *= weights[:-1]

        val = torch.mean(separate_losses)
        self.value = val.item()

        return val

    def reinitialize(self, mu):
        normalization = self.normalization
        epsilon = self.epsilon
        return Weighted_PDNN_Loss(self.ronn, normalization, mu, epsilon)
