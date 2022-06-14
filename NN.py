"""
TO DO:

2. Model training checkpoints (and also save current loss & validation loss
    so that we can choose the checkpoint based on validation)
        There is possibly more information we want to save than we are currently saving.
        We probably want to save the training set/training indices (and validation)

5. When reduction method is ReducedBasis, we currently aren't using a lot of training data
    (just the mu's from mu greedy which is very few-->high error on validation)
   Make it possible to use different training data for PDNN and PINN.

6. Make output normalization consistent between the types of loss functions.

7. Make error analysis by network into a better table like this (maybe use a new data structure):
                     Mean Relative Error
            NN-HF           NN-RO       RO-HF
    PINN
    PDNN
    PRNN
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
import matplotlib.pyplot as plt
from mlnics.Normalization import IdentityNormalization
from mlnics.Losses import reinitialize_loss
import sys


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

        self.train_idx = None
        self.validation_idx = None

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
        return torch.tensor(np.array(self.reduced_problem._combine_all_inner_products()))

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
                net_output = self.forward(input_normalization(mu_t[i], axis=0)).view(-1, 1).double()
                OV = OnlineVector(self.component_counts)
                OV.content = output_normalization(net_output, normalize=False).view(-1).detach().numpy()
                OF = OnlineFunction(OV)
                TS.append(OF)
            return TS
        else:
            net_output = self.forward(input_normalization(mu.view(1, -1), axis=0)).view(-1, 1).double()
            OV = OnlineVector(self.component_counts)
            OV.content = output_normalization(net_output, normalize=False).view(-1).detach().numpy()
            OF = OnlineFunction(OV)
            return OF

    def load(filename):
        """
        Load the neural network weights from a file.
        """
        raise NotImplementedError()


"""
Functions for getting test/validation sets
"""

def get_test(ronn):
    """
    Assumes that testing set has already been initialized.
    Returns: torch.tensor
    """
    mu = torch.tensor(ronn.reduction_method.testing_set)
    return mu


def normalize_and_train(ronn, data, loss_fn, input_normalization=None, optimizer=torch.optim.Adam,
                        epochs=10000, lr=1e-3, print_every=100, folder='./model_checkpoints/', use_validation=True):
    """
    If input_normalization has not yet been fit, then this function fits it to the training data.
    """
    assert data.initialized

    # default initialization for input_normalization
    if input_normalization is None:
        input_normalization = IdentityNormalization()

    train, validation = data.train_validation_split()
    train_normalized = input_normalization(train, axis=0) # also initializes normalization
    validation_normalized = input_normalization(validation, axis=0)

    # initialize the same loss as loss_fn for validation
    if validation is not None:
        val_t0_idx = data.get_validation_initial_time_index()
        train_t0_idx = data.get_train_initial_time_index()
        assert (val_t0_idx is None and train_t0_idx is None) or ronn.time_dependent

        val_snapshot_idx = data.get_validation_snapshot_index()
        train_snapshot_idx = data.get_train_snapshot_index()

        loss_fn_validation = reinitialize_loss(ronn, loss_fn, validation, val_snapshot_idx, val_t0_idx)

        if not loss_fn.operators_initialized:
            loss_fn.set_snapshot_index(train_snapshot_idx)
            loss_fn.set_initial_time_index(train_t0_idx)
            loss_fn.set_mu(train)

    ############################# Train the model #############################
    optimizer = optimizer(ronn.parameters(), lr=lr)

    optimizer.zero_grad()
    for e in range(epochs):
        coeff_pred = ronn(train_normalized)
        loss = loss_fn(coeff_pred, normalized_mu=train_normalized)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % print_every == 0:
            ronn.eval()
            if validation is not None and use_validation:
                pred = ronn(validation_normalized)
                validation_loss = loss_fn_validation(pred, normalized_mu=validation_normalized)
                print(e, loss.item(), f"\tLoss(validation) = {validation_loss.item()}")
                """
                torch.save({
                    'epoch': e,
                    'model_state_dict': ronn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'validation_loss': validation_loss
                }, folder + f"model_checkpoint_{e}")
                """
            else:
                print(e, loss.item())
                """
                torch.save({
                    'epoch': e,
                    'model_state_dict': ronn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, folder + f"model_checkpoint_{e}")
                """

            ronn.train()

    optimizer.zero_grad()

    return loss_fn_validation


def compute_reduced_solutions(reduced_problem, mu):
    """
    Compute the high fidelity solutions for each value of mu.
    mu: torch.tensor
    """
    solutions = []

    if "T" in dir(reduced_problem): # time dependent
        for m in mu:
            reduced_problem.set_mu(tuple(np.array(m)))
            reduced_problem.solve()
            for sol_t in reduced_problem._solution_over_time:
                sol_t = np.array(sol_t.vector()).reshape(-1, 1)
                solutions.append(sol_t)
    else:
        for m in mu:
            reduced_problem.set_mu(tuple(np.array(m)))
            reduced_problem.solve()
            sol = np.array(reduced_problem._solution.vector()).reshape(-1, 1)
            solutions.append(sol)

    return np.array(solutions)


def compute_error(ronn, pred, ro_solutions, euclidean=False):
    """
    Compute error based on the inner product for the given problem.
    Assumes that input normalization has already been applied to normalized_mu.
    hf_solutions are the high fidelity solutions corresponding to the non-normalized mu.
    """

    if euclidean or len(ronn.problem.components) == 1:
        errors = []
    else:
        errors = dict()

    for i in range(pred.shape[0]):
        ro_solution_i = ro_solutions[i]
        ro_solution_prediction_i = pred[i].reshape(-1, 1)
        difference = ro_solution_i - ro_solution_prediction_i

        if euclidean:
            norm_sq_diff = np.sum(difference**2)
            norm_sq_sol = np.sum(ro_solution_i**2)
            relative_error = np.sqrt(norm_sq_diff / norm_sq_sol)
            errors.append(relative_error)
        else:
            if len(ronn.problem.components) == 1:
                inner = ronn.problem.inner_product[0].array()
                norm_sq_diff = (difference.T @ inner @ difference)[0, 0]
                norm_sq_sol = (ro_solution_i.T @ inner @ ro_solution_i)[0, 0]
                relative_error = np.sqrt(norm_sq_diff / norm_sq_sol)
                errors.append(relative_error)
            else:
                for component in ronn.problem.components:
                    inner = ronn.problem.inner_product[component][0].array()
                    norm_sq_diff = (difference.T @ inner @ difference)[0, 0]
                    norm_sq_sol = (ro_solution_i.T @ inner @ ro_solution_i)[0, 0]
                    relative_error = np.sqrt(norm_sq_diff / norm_sq_sol)

                    if component not in errors:
                        errors[component] = [relative_error]
                    else:
                        errors[component].append(relative_error)

    return errors

def plot_error(ronn, mu, input_normalization=None, ind1=0, ind2=1, cmap="bwr"):
    """
    mu is not normalized and not time augmented
    """
    if input_normalization is None:
        input_normalization = IdentityNormalization()

    hf_solutions = compute_reduced_solutions(ronn.problem, mu)
    normalized_mu = input_normalization(mu)
    pred = ronn(normalized_mu).detach().numpy()

    coeff = ronn.get_coefficient_matrix().detach().numpy()
    errors = compute_error(ronn, (coeff @ pred.T).T, hf_solutions)
    plot = plt.scatter(mu[:, ind1], mu[:, ind2], c=errors, cmap=cmap)
    return errors, plot


def error_analysis_fixed_net(ronn, mu, input_normalization, euclidean=False, print_results=True):
    """
    mu: calculate error of neural network ronn on test set mu (not time augmented)
    """

    # get neural network predictions
    if ronn.time_dependent:
        normalized_mu = input_normalization(ronn.augment_parameters_with_time(mu), axis=0)
    else:
        normalized_mu = input_normalization(mu, axis=0)

    nn_solutions = ronn(normalized_mu).detach().numpy()
    ro_solutions = compute_reduced_solutions(ronn.reduced_problem, mu)
    hf_solutions = compute_reduced_solutions(ronn.problem, mu)

    # compute error from neural net to high fidelity
    coeff = ronn.get_coefficient_matrix().detach().numpy()
    nn_hf_error = compute_error(ronn, (coeff @ nn_solutions.T).T, hf_solutions, euclidean=euclidean)

    # compute error from neural net to reduced order
    nn_ro_error = compute_error(ronn, (coeff @ nn_solutions.T).T, coeff @ ro_solutions, euclidean=euclidean)

    # compute error from reduced order to high fidelity
    ro_hf_error = compute_error(ronn, (coeff @ ro_solutions.T).T, hf_solutions, euclidean=euclidean)

    if print_results:
        if len(ronn.problem.components) > 1:
            raise NotImplementedError("Error analysis for multi-component problems not implemented yet.")
        print("Mean Relative Error:")
        print("N\tNN-HF\t\t\tNN-RO\t\t\tRO-HF")
        print(f"{ronn.ro_dim}\t{np.mean(nn_hf_error)}\t{np.mean(nn_ro_error)}\t{np.mean(ro_hf_error)}")

    # perhaps return the solutions too so that they don't have to be computed every time this is called
    return nn_hf_error, nn_ro_error, ro_hf_error


def error_analysis_by_network(nets, mu, input_normalization, euclidean=False, print_results=True):
    """
    nets: dictionary of neural networks
    """
    for net_name in nets:
        net = nets[net_name]
        print("Error analysis for " + net_name)
        _ = error_analysis_fixed_net(net, mu, input_normalization, euclidean=euclidean, print_results=print_results)
        print("")

def error_analysis(ronn, mu, input_normalization, n_hidden=2, n_neurons=100, activation=torch.tanh):

    max_dim = ronn.ro_dim

    for dim in range(1, max_dim+1):
        # initialize RONN with dimension dim
        net = RONN(
            ronn.problem, ronn.reduction_method, n_hidden, n_neurons, activation
        )

        # train RONN

        # calculate error using error_analysis_fixed_net

        # print error for given dim ?
        pass



    raise NotImplementedError()



"""
Table for error analysis (perhaps need two types of tables?
--1 for net comparison, 1 for # basis functions comparison?)
"""
class RONNPerformanceTable:
    def __init__(self):
        pass

    def __str__(self):
        pass
