"""
TO DO:

1. When reduction method is ReducedBasis, we currently aren't using a lot of training data
    (just the mu's from mu greedy which is very few-->high error on validation)
   Make it possible to use different training data for PDNN and PINN.

2. Other tutorials

3. Speedup analysis

4. Batching

5. Boundary conditions in PINN loss?
"""

import torch
import torch.nn as nn
import numpy as np
from fenics import *
from rbnics import *
from rbnics.backends.online import OnlineFunction, OnlineVector
from rbnics.backends.common.time_series import TimeSeries
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.dolfin.evaluate import evaluate
from rbnics.utils.io.online_size_dict import OnlineSizeDict
from mlnics.Normalization import IdentityNormalization


class RONN(nn.Module):
    """
    Reduced Order Neural Network
    """
    def __init__(self, problem, reduction_method, n_hidden=2, n_neurons=100, activation=torch.tanh):
        """
        REQUIRES:
            problem.set_mu_range(...) has been called
        """

        super(RONN, self).__init__()
        self.name = "RONN"

        self.problem = problem
        self.reduction_method = reduction_method
        self.reduced_problem = reduction_method.reduced_problem

        ########################## load training set ##########################
        if 'POD' in dir(reduction_method):
            self.mu = torch.tensor(reduction_method.training_set)
        else:
            # greedy method
            reduction_method.greedy_selected_parameters.load(reduction_method.folder["post_processing"], "mu_greedy")
            self.mu = torch.tensor(reduction_method.greedy_selected_parameters)[:-1]

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
        self.layers = nn.ModuleList()
        last_n = self.num_params
        for i in range(n_hidden):
            self.layers.append(nn.Linear(last_n, n_neurons))
            last_n = n_neurons
        self.layers.append(nn.Linear(last_n, self.ro_dim))

        self.activation = activation

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
        S = torch.empty((self.ro_dim, self.num_snapshots))

        if self.time_dependent:
            for i in range(0, self.num_snapshots, self.num_times):
                self.problem.import_solution(self.problem.folder_prefix + "/snapshots", f"truth_{i//self.num_times}")
                snapshots = np.array([
                    s.vector() for s in self.reduced_problem.project(self.problem._solution_over_time)
                ]).T
                S[:, i:i+self.num_times] = torch.tensor(snapshots)

        else:
            for i in range(self.num_snapshots):
                self.problem.import_solution(self.problem.folder_prefix + "/snapshots", f"truth_{i}")
                snapshot = self.reduced_problem.project(self.problem._solution)
                S[:, i] = torch.tensor(np.array(snapshot.vector()))

        return S.double()

    def get_coefficient_matrix(self):
        if self.num_components == 1:
            coeff_matrix =  torch.tensor([
                coeff.vector() for coeff in self.reduced_problem.basis_functions
            ]).T.double()
        else:
            coefficients = []
            for component in self.problem.components:
                for c in self.reduced_problem.basis_functions[component]:
                    coefficients.append(c.vector())
            coeff_matrix = torch.tensor(coefficients).T.double()
        return coeff_matrix

    def get_inner_product_matrix(self):
        inner_product = torch.tensor(np.array(self.reduced_problem._combine_all_inner_products()))
        return inner_product

    def get_operator_matrices(self, mu=None):
        if mu is None:
            mu = self.mu if not self.time_dependent else self.time_augmented_mu

        operator_dict = dict()

        coeff = self.get_coefficient_matrix().detach().numpy()
        inner_prod = self.get_inner_product_matrix().detach().numpy()
        projection = (coeff @ inner_prod).T

        for term in self.problem.terms:
            # matrix terms
            if term in ['a', 'm', 'b', 'bt']:
                A = np.zeros((mu.shape[0], self.ro_dim, self.ro_dim))
                num_operators = len(self.problem.operator[term])
                operators = np.zeros((num_operators, self.ro_dim, self.ro_dim))
                for j, Aj in enumerate(self.problem.operator[term]):
                    if type(Aj) is ParametrizedTensorFactory:
                        Aj = np.array(evaluate(Aj).array())
                    else:
                        Aj = Aj.array()
                    operators[j] = projection @ Aj @ coeff

                for i, m in enumerate(mu):
                    self.problem.set_mu(tuple(np.array(m)[self.time_dependent:]))
                    thetas = np.array(self.problem.compute_theta(term)).reshape(-1, 1, 1)
                    A[i] = np.sum(thetas * operators, axis=0)
                A = torch.tensor(A).double()

                operator_dict[term] = A

            # vector terms
            elif term in ['c', 'f', 'g']:
                C = np.zeros((mu.shape[0], self.ro_dim))
                num_operators = len(self.problem.operator[term])
                operators = np.zeros((num_operators, self.ro_dim))
                for j, Cj in enumerate(self.problem.operator[term]):
                    if type(Cj) is ParametrizedTensorFactory:
                        Cj = np.array(evaluate(Cj))
                    else:
                        Cj = np.array(Cj)
                    operators[j] = np.matmul(projection, Cj.reshape(-1, 1)).reshape(-1)

                for i, m in enumerate(mu):
                    self.problem.set_mu(tuple(np.array(m)[self.time_dependent:]))
                    thetas = np.array(self.problem.compute_theta(term)).reshape(-1, 1)
                    C[i] = np.sum(thetas * operators, axis=0)
                C = torch.tensor(C).double()[:, :, None]

                operator_dict[term] = C

            else:
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