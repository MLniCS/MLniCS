import torch
import torch.nn as nn
import numpy as np
from dolfin import *
from rbnics import *
from rbnics.backends.basic.wrapping.delayed_transpose import DelayedTranspose
from rbnics.backends.online import OnlineFunction, OnlineVector
from rbnics.backends.common.time_series import TimeSeries
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends import evaluate, transpose
from rbnics.utils.io.online_size_dict import OnlineSizeDict
from mlnics.Normalization import IdentityNormalization

import time

NN_FOLDER = "/nn_results"

class RONN(nn.Module):
    """
    RONN (Reduced Order Neural Network) Class

    This class is a reduced order model that uses a neural network to map the high-dimensional parameters to the low-dimensional reduced order coefficients. It inherits from PyTorch's nn.Module class.

    Args:
    loss_type (str): Type of loss function used for training the neural network.
    problem (object): Problem instance for which the reduced order model is being created.
    reduction_method (object): Reduction method used for reducing the high-dimensional model to the low-dimensional reduced order model.
    n_hidden (int, optional): Number of hidden layers in the neural network. Defaults to 2.
    n_neurons (int, optional): Number of neurons in each hidden layer. Defaults to 100.
    activation (function, optional): Activation function used in each hidden layer. Defaults to torch.tanh.

    Attributes:
    VERBOSE (bool): Boolean flag to indicate if verbose output is desired.
    loss_type (str): Type of loss function used for training the neural network.
    problem (object): Problem instance for which the reduced order model is being created.
    reduction_method (object): Reduction method used for reducing the high-dimensional model to the low-dimensional reduced order model.
    reduced_problem (object): Reduced problem instance.
    mu (torch.tensor): Tensor of parameters used for training the neural network.
    space_dim (int): Dimension of the high-fidelity space.
    num_components (int): Number of components in the solution of the problem.
    ro_dim (int): Dimension of the reduced order space.
    component_counts (dict or OnlineSizeDict): Dictionary mapping each component to its count in the reduced order space.
    num_params (int): Number of parameters in the model.
    num_times (int): Number of times in the time-dependent problem.
    time_dependent (bool): Boolean flag to indicate if the problem is time-dependent.
    Tf (float): Final time in the time-dependent problem.
    T0 (float): Initial time in the time-dependent problem.
    dt (float): Time step in the time-dependent problem.
    time_augmented_mu (torch.tensor): Tensor of parameters used for training the neural network, augmented with time in the case of a time-dependent problem.
    num_snapshots (int): Number of snapshots in the training set.
    num_hidden (int): Number of hidden layers in the neural network.
    num_neurons (int): Number of neurons in each hidden layer.
    layers (nn.ModuleList): List of nn.Linear layers in the neural network.
    activation (function): Activation function used in each hidden layer.
    projection (None or nn.Linear): Linear layer used for projecting the reduced order coefficients back to the high-fidelity space.
    proj_snapshots (None or torch.tensor): Tensor of high-fidelity snapshots projected from the reduced order coefficients.

    Methods:
    forward(mu): Map the input parameter mu to the reduced order coefficient.
    name(): Return the name of the reduced order model.
    augment_parameters_with_time(mu): Augment the input parameter mu with time in the case of a time-dependent problem.
    get_projected_snap
    """

    def __init__(self, loss_type, problem, reduction_method, n_hidden=2, n_neurons=100, activation=torch.tanh):
        """
        REQUIRES:
            problem.set_mu_range(...) has been called
        """

        super(RONN, self).__init__()
        self.VERBOSE = True
        self.loss_type = loss_type

        self.problem = problem
        self.reduction_method = reduction_method
        self.reduced_problem = reduction_method.reduced_problem

        ########################## load training set ##########################
        if 'POD' in dir(reduction_method):
            self.mu = torch.tensor(reduction_method.training_set, dtype=torch.float64)
        else:
            # greedy method
            reduction_method.greedy_selected_parameters.load(reduction_method.folder["post_processing"], "mu_greedy")
            self.mu = torch.tensor(reduction_method.greedy_selected_parameters, dtype=torch.float64)[:-1]

        # dimension of high fidelity space
        self.space_dim = problem.V.dim()

        ############### compute dimension of reduced order space ###############
        self.num_components = len(problem.components) # number of components (e.g. velocity, pressure, etc.) of solution
        if self.num_components == 1:
            self.ro_dim = len(self.reduced_problem.basis_functions)
            self.component_counts = self.ro_dim
        elif self.num_components > 1:
            self.ro_dim = 0
            self.component_counts = OnlineSizeDict()
            for component in problem.components:
                self.component_counts[component] = len(self.reduced_problem.basis_functions[component])
                self.ro_dim += self.component_counts[component]
        else:
            raise ValueError("Number of components in problem must be >= 1")

        self.num_params = len(problem.mu_range) # number of parameters in model

        ########## adjust/add parameters for time dependent problems ##########
        self.num_times = 1
        self.time_dependent = False
        if "T" in dir(problem):
            self.time_dependent = True
            self.Tf = problem.T
            self.T0 = problem.t0
            self.dt = problem.dt

            # round to nearest int in case of roundoff errors
            self.num_times += int((self.Tf - self.T0) / self.dt + 0.5)

            # make time a new parameter in the model
            self.num_params += 1
            self.time_augmented_mu = self.augment_parameters_with_time(self.mu)

            self.num_snapshots = self.time_augmented_mu.shape[0]

        else:
            self.num_snapshots = self.mu.shape[0]

        #### store the network topology used in RONN.forward in self.layers ####
        self.num_hidden = n_hidden
        self.num_neurons = n_neurons
        self.layers = nn.ModuleList()
        last_n = self.num_params
        for i in range(n_hidden):
            self.layers.append(nn.Linear(last_n, n_neurons))
            last_n = n_neurons
        self.layers.append(nn.Linear(last_n, self.ro_dim))

        self.activation = activation

        self.projection = None
        self.proj_snapshots = None

    def name(self):
        return self.loss_type + "_" + str(self.num_hidden) + "_hidden_layers_" + str(self.num_neurons) + "_neurons"

    def forward(self, mu):
        """
        Map parameter mu --> reduced order coefficient
        """

        res = mu
        for layer in self.layers[:-1]:
            res = self.activation(layer(res))
        res = self.layers[-1](res)
        return res

    def augment_parameters_with_time(self, mu):
        if not self.time_dependent:
            return mu
        new_mu = torch.zeros((mu.shape[0]*self.num_times, self.num_params))
        for i in range(mu.shape[0]):
            for j in range(self.num_times):
                new_mu[i*self.num_times+j, 1:] = mu[i]
                new_mu[i*self.num_times+j, 0] = self.T0 + j * self.dt
        return new_mu

    def get_projected_snapshots(self):
        if self.proj_snapshots is None:
            S = torch.empty((self.ro_dim, self.num_snapshots))

            if self.time_dependent:
                for i in range(0, self.num_snapshots, self.num_times):
                    self.problem.import_solution(self.problem.folder_prefix + "/snapshots", f"truth_{i//self.num_times}")
                    snapshots = np.array([
                        s.vector() for s in self.reduced_problem.project(self.problem._solution_over_time)
                    ]).T
                    S[:, i:i+self.num_times] = torch.tensor(snapshots, dtype=torch.float64)

            else:
                for i in range(self.num_snapshots):
                    self.problem.import_solution(self.problem.folder_prefix + "/snapshots", f"truth_{i}")
                    snapshot = self.reduced_problem.project(self.problem._solution)
                    S[:, i] = torch.tensor(np.array(snapshot.vector(), dtype=np.float64), dtype=torch.float64)

            self.proj_snapshots = S.double()

        return self.proj_snapshots

    def get_coefficient_matrix(self):
        if self.num_components == 1:
            coeff_matrix =  torch.tensor([
                coeff.vector() for coeff in self.reduced_problem.basis_functions
            ], dtype=torch.float64).T
        else:
            coefficients = []
            for component in self.problem.components:
                for c in self.reduced_problem.basis_functions[component]:
                    coefficients.append(c.vector())
            coeff_matrix = torch.tensor(coefficients, dtype=torch.float64).T
        return coeff_matrix

    def get_inner_product_matrix(self):
        inner_product = torch.tensor(np.array(self.reduced_problem._combine_all_inner_products()))
        return inner_product

    def get_reduced_operator_matrices(self, mu=None):
        if mu is None:
            mu = self.mu if not self.time_dependent else self.time_augmented_mu

        operator_dict = dict()

        if self.projection is None:
            coeff = self.get_coefficient_matrix().detach().numpy()
            inner_prod = self.get_inner_product_matrix().detach().numpy()
            projection = (coeff @ inner_prod).T
            self.projection = projection
        else:
            projection = self.projection

        for term in self.reduced_problem.terms:
            # matrix terms
            if term in ['a', 'm', 'b', 'bt']:
                A = np.zeros((mu.shape[0], self.ro_dim, self.ro_dim))
                num_operators = len(self.reduced_problem.operator[term])
                operators = np.zeros((num_operators, self.ro_dim, self.ro_dim))
                for j, Aj in enumerate(self.reduced_problem.operator[term]):
                    if type(Aj) is ParametrizedTensorFactory:
                        Aj = np.array(evaluate(Aj).array())
                    elif type(Aj) is DelayedTranspose:
                        Aj = np.array([v.vector() for v in Aj._args[0]]) @ np.array(evaluate(Aj._args[1]).array()) @ np.array([v.vector() for v in Aj._args[2]]).T
                    else:
                        Aj = Aj.reshape(-1)[0].content
                    operators[j] = Aj
                
                if self.time_dependent:
                    rp_time = self.reduced_problem.t
                for i, m in enumerate(mu):
                    self.reduced_problem.set_mu(tuple(np.array(m)[self.time_dependent:]))
                    if self.time_dependent:
                        self.reduced_problem.set_time(np.array(m)[0])
                    thetas = np.array(self.reduced_problem.compute_theta(term)).reshape(-1, 1, 1)
                    A[i] = np.sum(thetas * operators, axis=0)
                if self.time_dependent:
                    self.reduced_problem.set_time(rp_time)
                A = torch.tensor(A, dtype=torch.float64)

                operator_dict[term] = A
                
            elif term in ['f', 'g']:
                C = np.zeros((mu.shape[0], self.ro_dim))
                num_operators = len(self.reduced_problem.operator[term])
                operators = np.zeros((num_operators, self.ro_dim))
                for j, Cj in enumerate(self.reduced_problem.operator[term]):
                    if type(Cj) is ParametrizedTensorFactory:
                        Cj = np.array(evaluate(Cj))
                    elif type(Cj) is DelayedTranspose:
                        Cj = np.array(transpose(Cj._args[0]) * evaluate(Cj._args[1]))
                    else:
                        Cj = Cj.reshape(-1)[0].content
                    operators[j] = Cj
                
                if self.time_dependent:
                    rp_time = self.reduced_problem.t
                for i, m in enumerate(mu):
                    self.reduced_problem.set_mu(tuple(np.array(m)[self.time_dependent:]))
                    if self.time_dependent:
                        self.reduced_problem.set_time(np.array(m)[0])
                    thetas = np.array(self.reduced_problem.compute_theta(term)).reshape(-1, 1)
                    C[i] = np.sum(thetas * operators, axis=0)
                if self.time_dependent:
                    self.reduced_problem.set_time(rp_time)
                C = torch.tensor(C, dtype=torch.float64)[:, :, None]                
                operator_dict[term] = C

            else:
                if self.VERBOSE:
                    print(f"Operator '{term}' not implemented. Continuing without operator '{term}'...")

        return operator_dict

    """ make this work more like rbnics solve function for consistency """
    def solve(self, mu, input_normalization=None, output_normalization=None):
        assert len(mu.shape) == 1

        if input_normalization is None:
            input_normalization = IdentityNormalization()

        if output_normalization is None:
            output_normalization = IdentityNormalization()

        if self.time_dependent:
            # create time series
            TS = TimeSeries((self.T0, self.Tf), self.dt)
            mu_t = self.augment_parameters_with_time(mu.view(1, -1))
            for i, t in enumerate(np.linspace(self.T0, self.Tf, self.num_times)):
                net_output = self.forward(input_normalization(mu_t[i])).view(-1, 1).double()
                OV = OnlineVector(self.component_counts)
                OV.content = output_normalization(net_output, normalize=False).view(-1).detach().numpy()
                OF = OnlineFunction(OV)
                TS.append(OF)
            return TS
        else:
            net_output = self.forward(input_normalization(mu.view(1, -1))).view(-1, 1).double()
            OV = OnlineVector(self.component_counts)
            OV.content = output_normalization(net_output, normalize=False).view(-1).detach().numpy()
            OF = OnlineFunction(OV)
            return OF

    def load_best_weights(self):
        folder = self.reduction_method.folder_prefix + NN_FOLDER + "/" + self.name()
        checkpoint = torch.load(folder + f"/checkpoint.pt")
        self.load_state_dict(checkpoint['model_state_dict'])
