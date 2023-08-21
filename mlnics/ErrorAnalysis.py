import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from dolfin import *
from rbnics import *
from rbnics.utils.io.text_line import TextLine
from mlnics.Normalization import IdentityNormalization
from mlnics.NN import RONN
from mlnics.Losses import PINN_Loss

NN_FOLDER = "/nn_results"

def compute_reduced_solutions(reduced_problem, mu):
    """
    Computes the high-fidelity solutions for each value of `mu`.
    
    Parameters:
        reduced_problem (Object): The reduced problem instance.
        mu (torch.tensor): A tensor containing the values of `mu` for which solutions need to be computed.
    
    Returns:
        numpy.ndarray: A numpy array containing the solutions.
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
    Compute the error between the predicted and actual high-fidelity solutions.
    It assumes that input normalization has already been applied to the normalized mu.
    The error can be calculated in either relative or absolute form, and can be computed using the Euclidean norm or the inner product.

    Parameters:
        ronn (object): The reduced order model object.
        pred (np.ndarray): The predicted high-fidelity solutions.
        ro_solutions (np.ndarray): The actual high-fidelity solutions.
        euclidean (bool, optional): If True, the error will be computed using the Euclidean norm, otherwise it will be computed using the inner product. Default is False.
        relative (bool, optional): If True, the error will be computed in relative form, otherwise it will be computed in absolute form. Default is True.
        eps (float, optional): A small number used to avoid division by zero in the relative error calculation. Default is 1e-12.

    Returns:
        list: A list of errors, one for each prediction.
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
    Plots the error of the reduced order neural network (ronn) predictions using data, mu and a specified color map.

    :param ronn: The reduced order neural network model.
    :type ronn: torch.nn.Module
    :param data: The data used for the model.
    :type data: object
    :param mu: The parameters for the ronn model.
    :type mu: np.ndarray
    :param input_normalization: The normalization method for the input data. Defaults to None.
    :type input_normalization: callable or None, optional
    :param output_normalization: The normalization method for the output data. Defaults to None.
    :type output_normalization: callable or None, optional
    :param ind1: The index of the first parameter to be plotted. Defaults to 0.
    :type ind1: int, optional
    :param ind2: The index of the second parameter to be plotted. Defaults to 1.
    :type ind2: int, optional
    :param cmap: The color map to be used for plotting. Defaults to "bwr".
    :type cmap: str, optional
    :return: The computed errors and the plot.
    :rtype: tuple (np.ndarray, matplotlib.collections.PathCollection)

    Note:
    mu is not normalized and not time augmented.
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
    Calculate the residuals of a reduced order neural network (RONN) based on input data and parameters `mu`.

    Parameters:
    ronn : reduced order neural network model
    The reduced order neural network model to be evaluated.
    data : data object
    The data object that contains the training data.
    mu : numpy array
    The parameters to be evaluated.
    input_normalization : callable or None, optional (default: None)
    The normalization function for the input parameters.
    If None, the `IdentityNormalization` function will be used.
    output_normalization : callable or None, optional (default: None)
    The normalization function for the output data.
    loss_functions : list or None, optional (default: None)
    The list of loss functions to be used for evaluating the RONN.
    plot_residuals : bool, optional (default: True)
    Whether to plot the residuals and save the plot to disk.
    ind1 : int, optional (default: 0)
    The first index of the parameter array `mu` to be used in plotting.
    ind2 : int, optional (default: 1)
    The second index of the parameter array `mu` to be used in plotting.
    cmap : str, optional (default: "bwr")
    The color map to be used in plotting the residuals.

    Returns:
    losses : numpy array
    The array of residuals.
    plot : matplotlib.collections.PathCollection or None
    The plot of the residuals, if `plot_residuals` is True.
    loss_functions : list
    The list of loss functions used for evaluating the RONN.

    Notes
    - `mu` is not normalized and not time augmented.
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
    Analyze the error of a neural network on test set `mu` (not time augmented).

    Parameters:
    - `ronn` (object): A neural network object.
    - `mu` (array-like): Test set.
    - `input_normalization` (function): Function to normalize input data.
    - `output_normalization` (function): Function to normalize output data.
    - `euclidean` (bool, optional): Use euclidean distance for error computation. Defaults to False.
    - `relative` (bool, optional): Use relative error for computation. Defaults to True.
    - `print_results` (bool, optional): Print results to console. Defaults to True.

    Returns:
    - tuple of arrays: Three arrays containing errors of neural network compared to high-fidelity, neural network compared to reduced order, and reduced order compared to high-fidelity, respectively.
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
        print(TextLine(ronn.loss_type+" N = "+str(ronn.ro_dim), fill="#"))
        print(f"ERROR\tNN-HF\t\t\tNN-RO\t\t\tRO-HF")
        print(f"min\t{np.min(nn_hf_error)}\t{np.min(nn_ro_error)}\t{np.min(ro_hf_error)}")
        print(f"mean\t{np.mean(nn_hf_error)}\t{np.mean(nn_ro_error)}\t{np.mean(ro_hf_error)}")
        print(f"max\t{np.max(nn_hf_error)}\t{np.max(nn_ro_error)}\t{np.max(ro_hf_error)}")

    # perhaps return the solutions too so that they don't have to be computed every time this is called
    return nn_hf_error, nn_ro_error, ro_hf_error


def error_analysis_by_network(nets, mu, input_normalizations, output_normalizations, euclidean=False, relative=True):
    """
    This function calculates the error of the neural network `ronn` on the test set `mu` (not time augmented). 
        
    :param ronn: The neural network to be evaluated.
    :param mu: The test set to be used for error evaluation.
    :param input_normalization: Normalization function applied to the input data.
    :param output_normalization: Normalization function applied to the output data.
    :param euclidean: Boolean indicating whether to use euclidean error calculation. Defaults to False.
    :param relative: Boolean indicating whether to use relative error calculation. Defaults to True.
    :param print_results: Boolean indicating whether to print the results. Defaults to True.
    :return: Tuple containing the errors between (1) neural network and high-fidelity, (2) neural network and reduced-order, and (3) reduced-order and high-fidelity.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """

    print(85*"#")

    for i, net_name in enumerate(nets):
        net = nets[net_name]
        input_normalization = input_normalizations[net_name]
        output_normalization = output_normalizations[net_name]
        if i == 0:
            if relative:
                print(f"Mean Relative Error for N = {net.ro_dim} Basis Functions")
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


def plot_solution(ronn, mu, input_normalization=None, output_normalization=None, t=-1, colorbar=True, component=-1):
    """
    Plots the solution of a reduced order neural network model.

    Parameters:
    ronn (object): An instance of the reduced order neural network class.
    mu (tuple): A tuple representing the parameter values for the model.
    input_normalization (None, optional): The normalization applied to the inputs. Defaults to None.
    output_normalization (None, optional): The normalization applied to the outputs. Defaults to None.
    t (int, optional): The time step for a time dependent model. Defaults to 0.
    colorbar (bool, optional): Flag to indicate if a colorbar should be displayed with the plot. Defaults to True.
    component (int, optional): The component number to plot. Defaults to -1.

    Returns:
    None
    """
    problem = ronn.problem

    problem.set_mu(mu)
    problem.solve()

    if not ronn.time_dependent:
        if component != -1:
            P = plot(problem._solution, component=component)
        else:
            P = plot(problem._solution)
    else:
        if component != -1:
            P = plot(problem._solution_over_time[t], component=component)
        else:
            P = plot(problem._solution_over_time[t])

    if colorbar:
        cbar = plt.colorbar(P)
        tick_locator = MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.tight_layout()

    plt.title("Solution field at $\mu$ = "+str(tuple(round(i, 2) for i in mu)))
    folder = ronn.reduction_method.folder_prefix + NN_FOLDER + "/" + ronn.name()
    plt.savefig(folder + "/" + ronn.name() + f"_{mu}_solution.png")


def plot_solution_difference(ronn, mu, input_normalization=None, output_normalization=None, t=-1, colorbar=True, component=-1):
    """
    Plot the difference between the solution of the original problem and the reduced neural network solution.
    
    Parameters:
    ronn : object
    Instance of the reduced neural network.
    mu : tuple
    The parameter of the PDE.
    input_normalization : float, optional
    Normalization factor for the input, by default None
    output_normalization : float, optional
    Normalization factor for the output, by default None
    t : int, optional
    Time step for time-dependent problems, by default 0
    colorbar : bool, optional
    Show colorbar, by default True
    component : int, optional
    The component of the solution to be plotted, by default -1 (all components)
    """

    problem = ronn.problem
    reduced_problem = ronn.reduced_problem
    V = problem.V

    mu_nn = torch.tensor(mu, dtype=torch.float64)
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
        else:
            P = plot(
                    project(
                        problem._solution\
                            - reduced_problem.basis_functions * nn_solution, V
                    )
            )
    else:
        if component != -1:
            P = plot(
                    project(
                        problem._solution_over_time[t]\
                            - reduced_problem.basis_functions * nn_solution[t], V
                    ), component=component
            )
        else:
            P = plot(
                    project(
                        problem._solution_over_time[t]\
                            - reduced_problem.basis_functions * nn_solution[t], V
                    )
            )

    if colorbar:
        cbar = plt.colorbar(P)
        tick_locator = MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.tight_layout()

    plt.title("Solution difference at $\mu$ = "+str(tuple(round(i, 2) for i in mu)))
    folder = ronn.reduction_method.folder_prefix + NN_FOLDER + "/" + ronn.name()
    plt.savefig(folder + "/" + ronn.name() + f"_{mu}_solution_difference.png")
