"""
TO DO:

1. Validation set code
2. Model training checkpoints (and also save current loss & validation loss
    so that we can choose the checkpoint based on validation)
        There is possibly more information we want to save than we are currently saving.

3. reduced_problem.project has an argument called "on_dirichlet_bc=False". What to do with this?
4. Maybe remove all of the mu_old stuff because the other fields of the problem aren't also being saved
    and may also change.

5. When reduction method is ReducedBasis, we currently aren't using a lot of training data
    (just the mu's from mu greedy which is very few-->high error on validation)
6. I think in PDNN Loss we need to normalize the output at each calculation of the loss
    because otherwise PRNN Loss won't work (PINN doesn't normalize output, PDNN does)
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
            self.time_augmented_mu = self.time_augment_parameter(self.mu)

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

    """ perhaps change name to 'augment_parameters_with_time' """
    def time_augment_parameter(self, mu):
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

    def get_operator_matrices(self):
        mu_old = self.problem.mu
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

        self.problem.set_mu(mu_old)

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
            mu_t = self.time_augment_parameter(mu.view(1, -1))
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
    #if ronn.time_dependent:
    #    mu = ronn.time_augment_parameter(mu)
    return mu

def get_test_validation(ronn, validation_proportion=0.2):
    """
    Returns: torch.tensor, torch.tensor/None
    The second return value is None if and only if the number of validation examples is 0
    """
    test_set = get_test(ronn)
    num_validation = int(test_set.shape[0]*validation_proportion)
    if num_validation == 0:
        return test_set, None
    else:
        perm = torch.randperm(test_set.shape[0])
        validation_idx = perm[:num_validation]
        test_idx = perm[num_validation:]
        test, validation = test_set[test_idx], test_set[validation_idx]
        return test, validation

"""
Functions for training and testing
"""

""" perhaps we want to require input_normalization and make a new function 'train' without any option for input normalization """
def normalize_and_train(ronn, loss_fn, input_normalization=None, optimizer=torch.optim.Adam,
                        epochs=10000, lr=1e-3, validation_set=None, print_every=100, folder='./model_checkpoints'):
    """
    If input_normalization has not yet been fit, then this function fits it to the training data.
    If not None, validation_set must not have been normalized.
    validation_set does not contain time as a parameter.
    """

    # set default initialization to input_normalization
    if input_normalization is None:
        input_normalization = IdentityNormalization()

    # first call of input_normalization "trains" the normalization
    mu = ronn.mu if not ronn.time_dependent else ronn.time_augmented_mu
    normalized_mu = input_normalization(mu, axis=0)

    # compute high fidelity solutions for computing error for validation
    if validation_set is not None:
        assert validation_set.shape[0] > 0
        hf_solutions = compute_reduced_solutions(ronn.reduced_problem, validation_set)
        # normalize validation set
        normalized_validation_set = input_normalization(validation_set, axis=0)


    ############################# Train the model #############################
    optimizer = optimizer(ronn.parameters(), lr=lr)

    optimizer.zero_grad()
    for e in range(epochs):
        coeff_pred = ronn(normalized_mu)
        loss = loss_fn(coeff_pred, normalized_mu=normalized_mu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % print_every == 0:
            ronn.eval()
            if validation_set is not None:
                # we also know that validation_set has nonzero size
                pred = ronn(normalized_validation_set).detach().numpy()
                """ There will be problems with multiple components """
                #errors = compute_error(ronn, pred, hf_solutions)
                #validation_loss = loss_fn(pred, normalized_mu=normalized_validation_set)
                #print(e, loss.item(), f"\tLoss(validation) = {np.mean(errors)}")
                torch.save({
                    'epoch': e,
                    'model_state_dict': ronn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'validation_loss': validation_loss
                }, folder)
            else:
                print(e, loss.item())
                torch.save({
                    'epoch': e,
                    'model_state_dict': ronn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, folder)

            ronn.train()


    optimizer.zero_grad()




def compute_reduced_solutions(reduced_problem, mu):
    """
    Compute the high fidelity solutions for each value of mu.
    mu: torch.tensor
    """
    mu_old = reduced_problem.mu

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

    reduced_problem.set_mu(mu_old)

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
        normalized_mu = input_normalization(ronn.time_augment_parameter(mu), axis=0)
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
