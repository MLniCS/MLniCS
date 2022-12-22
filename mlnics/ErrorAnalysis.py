import torch
import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from rbnics import *
from rbnics.utils.io.text_line import TextLine
from mlnics.Normalization import IdentityNormalization
from mlnics.NN import RONN
from mlnics.Losses import PINN_Loss

NN_FOLDER = "/nn_results"

def compute_reduced_solutions(reduced_problem, mu):
    """
    Compute the high fidelity solutions for each value of mu.
    mu: torch.tensor
    """
    solutions = []

    if "T" in dir(reduced_problem): # time dependent
        for m in mu:
            reduced_problem.set_mu(tuple(np.array(m)))
            solution = reduced_problem.solve()
            for sol_t in solution:
                sol_t = np.array(sol_t.vector()).reshape(-1, 1)
                solutions.append(sol_t)
    else:
        for m in mu:
            reduced_problem.set_mu(tuple(np.array(m)))
            reduced_problem.solve()
            sol = np.array(reduced_problem._solution.vector()).reshape(-1, 1)
            solutions.append(sol)

    return np.array(solutions)


def compute_error(ronn, pred, ro_solutions, euclidean=False, relative=True, eps=1e-12):
    """
    Compute error based on the inner product for the given problem.
    Assumes that input normalization has already been applied to normalized_mu.
    hf_solutions are the high fidelity solutions corresponding to the non-normalized mu.
    """

    errors = []

    for i in range(pred.shape[0]):
        ro_solution_i = ro_solutions[i]
        ro_solution_prediction_i = pred[i].reshape(-1, 1)
        difference = ro_solution_i - ro_solution_prediction_i

        if euclidean:
            norm_sq_diff = np.sum(difference**2)
            norm_sq_sol = np.sum(ro_solution_i**2)
            if relative:
                relative_error = np.sqrt(norm_sq_diff / (norm_sq_sol + eps))
                errors.append(relative_error)
            else:
                error = np.sqrt(norm_sq_diff)
                errors.append(error)
        else:
            if len(ronn.problem.components) == 1:
                inner = ronn.problem.inner_product[0].array()
            else:
                inner = ronn.problem._combined_inner_product.array()
            norm_sq_diff = (difference.T @ inner @ difference)[0, 0]
            norm_sq_sol = (ro_solution_i.T @ inner @ ro_solution_i)[0, 0]
            if relative:
                relative_error = np.sqrt(norm_sq_diff / (norm_sq_sol + eps))
                errors.append(relative_error)
            else:
                error = np.sqrt(norm_sq_diff)
                errors.append(error)

    return errors

def plot_error(ronn, data, mu, input_normalization=None, output_normalization=None, ind1=0, ind2=1, cmap="bwr"):
    """
    mu is not normalized and not time augmented
    """
    if input_normalization is None:
        input_normalization = IdentityNormalization()
    if output_normalization is None:
        output_normalization = IdentityNormalization()

    hf_solutions = compute_reduced_solutions(ronn.problem, mu)
    normalized_mu = input_normalization(mu)
    pred = output_normalization(ronn(normalized_mu).T, normalize=False).detach().numpy().T

    coeff = ronn.get_coefficient_matrix().detach().numpy()
    errors = compute_error(ronn, (coeff @ pred.T).T, hf_solutions)
    plot = plt.scatter(mu[:, ind1], mu[:, ind2], c=errors, cmap=cmap)
    plt.colorbar()
    plt.scatter(data.train_data[:, ind1], data.train_data[:, ind2], color='g')

    folder = ronn.reduction_method.folder_prefix + NN_FOLDER + "/" + ronn.name()
    plt.savefig(folder + "/error_by_parameter.png")


    return errors, plot

def get_residuals(ronn, data, mu,
                  input_normalization=None, output_normalization=None,
                  loss_functions=None,
                  plot_residuals=True, ind1=0, ind2=1, cmap="bwr"):
    """
    mu is not normalized and not time augmented
    """

    if input_normalization is None:
        input_normalization = IdentityNormalization()

    losses = []
    if loss_functions is None:
        loss_functions = []

    for i, mu_ in enumerate(mu):
        mu_ = mu_.reshape(1, -1)
        normalized_mu = input_normalization(mu_)
        if len(loss_functions) < mu.shape[0]:
            pinn_loss = PINN_Loss(ronn, output_normalization, mu=mu_)
            loss_functions.append(pinn_loss)
        else:
            pinn_loss = loss_functions[i]

        pred = ronn(normalized_mu)
        loss = pinn_loss(prediction_no_snap=pred, input_normalization=input_normalization, normalized_mu=normalized_mu).item()
        losses.append(loss)

    if plot_residuals:
        plot = plt.scatter(mu[:, ind1], mu[:, ind2], c=losses, cmap=cmap)
        plt.colorbar()
        plt.scatter(data.train_data[:, ind1], data.train_data[:, ind2], color='g')

        folder = ronn.reduction_method.folder_prefix + NN_FOLDER + "/" + ronn.name()
        plt.savefig(folder + "/pinn_loss_by_parameter.png")


        return np.array(losses), plot, loss_functions
    else:
        return np.array(losses), None, loss_functions


def error_analysis_fixed_net(ronn, mu, input_normalization, output_normalization, euclidean=False, relative=True, print_results=True):
    """
    mu: calculate error of neural network ronn on test set mu (not time augmented)
    """

    # get neural network predictions
    if ronn.time_dependent:
        normalized_mu = input_normalization(ronn.augment_parameters_with_time(mu))
    else:
        normalized_mu = input_normalization(mu)

    nn_solutions = output_normalization(ronn(normalized_mu).T, normalize=False).T.detach().numpy()
    ro_solutions = compute_reduced_solutions(ronn.reduced_problem, mu)
    hf_solutions = compute_reduced_solutions(ronn.problem, mu)

    # compute error from neural net to high fidelity
    coeff = ronn.get_coefficient_matrix().detach().numpy()
    nn_hf_error = compute_error(ronn, (coeff @ nn_solutions.T).T, hf_solutions, euclidean=euclidean, relative=relative)

    # compute error from neural net to reduced order
    nn_ro_error = compute_error(ronn, (coeff @ nn_solutions.T).T, coeff @ ro_solutions, euclidean=euclidean, relative=relative)

    # compute error from reduced order to high fidelity
    ro_hf_error = compute_error(ronn, (coeff @ ro_solutions.T).T, hf_solutions, euclidean=euclidean, relative=relative)

    if print_results:
        print(TextLine("N = "+str(ronn.ro_dim), fill="#"))
        print(f"ERROR\tNN-HF\t\t\tNN-RO\t\t\tRO-HF")
        print(f"min\t{np.min(nn_hf_error)}\t{np.min(nn_ro_error)}\t{np.min(ro_hf_error)}")
        print(f"mean\t{np.mean(nn_hf_error)}\t{np.mean(nn_ro_error)}\t{np.mean(ro_hf_error)}")
        print(f"max\t{np.max(nn_hf_error)}\t{np.max(nn_ro_error)}\t{np.max(ro_hf_error)}")

    # perhaps return the solutions too so that they don't have to be computed every time this is called
    return nn_hf_error, nn_ro_error, ro_hf_error



def error_analysis_by_network(nets, mu, input_normalizations, output_normalizations, euclidean=False, relative=True):
    """
    nets: dictionary of neural networks
    """

    print(85*"#")

    for i, net_name in enumerate(nets):
        net = nets[net_name]
        input_normalization = input_normalizations[net_name]
        output_normalization = output_normalizations[net_name]
        if i == 0:
            if relative:
                print(f"Mean Relative Error for {net.ro_dim} Basis Functions")
            else:
                print(f"Mean Error for {net.ro_dim} Basis Functions")
            print("Network\t\tNN-HF\t\t\tNN-RO\t\t\tRO-HF")
        nn_hf_error, nn_ro_error, ro_hf_error = error_analysis_fixed_net(
            net, mu, input_normalization, output_normalization,
            euclidean=euclidean, relative=relative,
            print_results=False
        )
        print(f"{net_name}\t{np.mean(nn_hf_error)}\t{np.mean(nn_ro_error)}\t{np.mean(ro_hf_error)}")

    print(85*"#")

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

def plot_solution_difference(ronn, mu, input_normalization=None, output_normalization=None, t=0, colorbar=False, component=-1):
    """
    mu is a tuple.
    t is an int.
    component is a str.
    """
    problem = ronn.problem
    reduced_problem = ronn.reduced_problem
    V = problem.V

    mu_nn = torch.tensor(mu)
    nn_solution = ronn.solve(mu_nn, input_normalization, output_normalization)
    problem.set_mu(mu)
    problem.solve()

    if not ronn.time_dependent:
        if component != -1:
            P = plot(
                    project(
                        problem._solution\
                            - reduced_problem.basis_functions * nn_solution, V
                    ), component=component
            )
            if colorbar:
                plt.colorbar(P)
        else:
            P = plot(
                    project(
                        problem._solution\
                            - reduced_problem.basis_functions * nn_solution, V
                    )
            )
            if colorbar:
                plt.colorbar(P)
    else:
        if component != -1:
            P = plot(
                    project(
                        problem._solution_over_time[t]\
                            - reduced_problem.basis_functions * nn_solution[t], V
                    ), component=component
            )
            if colorbar:
                plt.colorbar(P)
        else:
            P = plot(
                    project(
                        problem._solution_over_time[t]\
                            - reduced_problem.basis_functions * nn_solution[t], V
                    )
            )
            if colorbar:
                plt.colorbar(P)

    folder = ronn.reduction_method.folder_prefix + NN_FOLDER + "/" + ronn.name()
    plt.savefig(folder + "/" + ronn.name() + f"_{mu}_solution_difference.png")
