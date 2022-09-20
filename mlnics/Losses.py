import torch
import numpy as np
from mlnics.Normalization import IdentityNormalization

from rbnics.backends.basic.wrapping.delayed_transpose import DelayedTranspose
from rbnics.backends.online import OnlineFunction, OnlineVector
from rbnics.backends.common.time_series import TimeSeries
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.dolfin.evaluate import evaluate

from dolfin import assemble

import time

class RONN_Loss_Base:
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
    PINN_Loss

    ronn: object of type RONN

    RETURNS: loss function loss_fn(parameters, reduced order coefficients)
    """
    def __init__(self, ronn, normalization=None, beta=1., mu=None):
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

    def name(self):
        return "PINN"

    def _compute_operators(self):
        self.operators_initialized = True

        #self.operators = self.ronn.get_operator_matrices(self.mu)
        self.operators = self.ronn.get_reduced_operator_matrices(self.mu)

        if self.time_dependent:
            self.T0_idx = torch.arange(0, self.mu.shape[0], self.ronn.num_times)

            self.T0_snapshots = torch.zeros((self.ronn.ro_dim, self.T0_idx.shape[0]))
            final_time = self.ronn.reduced_problem.T # store old final time to reset later
            self.ronn.reduced_problem.set_final_time(self.ronn.reduced_problem.t0)
            for i, mu in enumerate(self.mu[self.T0_idx]):
                self.ronn.reduced_problem.set_mu(tuple(np.array(mu)[1:]))
                solution = torch.Tensor(np.array(self.ronn.reduced_problem.solve()[0].vector())).view(-1, 1)
                self.T0_snapshots[:, [i]] = solution
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
            res1 -= self.operators['f']
        if 'c' in self.operators:
            for i, mu in enumerate(kwargs["input_normalization"](kwargs["normalized_mu"], normalize=False)):
                self.ronn.reduced_problem.set_mu(tuple(list(np.array(mu)[self.ronn.time_dependent:])))
                start_time = time.time()
                if self.ronn.time_dependent:
                    if i % self.ronn.num_times == 0:
                        TS = TimeSeries((self.ronn.T0, self.ronn.Tf), self.ronn.dt)
                        mu_t = self.ronn.augment_parameters_with_time(mu.view(1, -1))
                        for j, t in enumerate(np.linspace(self.ronn.T0, self.ronn.Tf, self.ronn.num_times)):
                            OV = OnlineVector(self.ronn.component_counts)
                            OV.content = self.normalization(net_output, normalize=False).view(-1).detach().numpy()
                            OF = OnlineFunction(OV)
                            TS.append(OF)
                        self.ronn.reduced_problem._solution_over_time = TS
                else:
                    net_output = pred[i].double().view(-1)
                    OV = OnlineVector(self.ronn.component_counts)
                    OV.content = net_output.detach().numpy()
                    OF = OnlineFunction(OV)
                    self.ronn.reduced_problem._solution = OF
                end_time = time.time()
                #print("If statement time:", end_time - start_time)

                #start_time = time.time()
                Basis_Matrix = np.array([v.vector()[:] for v in self.ronn.reduced_problem.basis_functions])
                self.ronn.problem.set_mu(tuple([e.item() for e in np.array(mu)[self.ronn.time_dependent:]]))
                thetas = self.ronn.problem.compute_theta('c')
                self.ronn.problem.solve()
                self.ronn.problem._solution.vector()[:] = (Basis_Matrix.T @ pred[i].detach().numpy().reshape(-1, 1)).reshape(-1)
                #print(self.ronn.problem.mu)
                operator_form = self.ronn.problem.assemble_operator('c')
                C = 0
                for j, Cj in enumerate(operator_form):
                    C += thetas[j] * np.array(assemble(Cj)[:]).reshape(-1, 1)
                C = Basis_Matrix @ C
                #operators = self.ronn.reduced_problem.assemble_operator('c')
                #end_time = time.time()
                #print("Assembly time:", end_time - start_time)
                #start_time = time.time()
                #thetas = np.array(self.ronn.reduced_problem.compute_theta('c'))
                #end_time = time.time()
                #print("Compute theta time:", end_time - start_time)

                #C = 0
                #start_time = time.time()
                #for j, Cj in enumerate(operators):
                #    if type(Cj) is ParametrizedTensorFactory:
                #        Cj = np.array(evaluate(Cj)).reshape(-1, 1)
                #    elif type(Cj) is DelayedTranspose:
                #        Cj = np.array([v.vector() for v in Cj._args[0]]) @ np.array(evaluate(Cj._args[1])).reshape(-1, 1)
                #    else:
                #        Cj = Cj.reshape(-1)[0].content.reshape(-1, 1)

                #    C += thetas[j] * Cj
                #end_time = time.time()
                #print("Numpyify operators time:", end_time - start_time)
                #print("")
                # set element of self.operators['c']
                self.operators['c'][i] = torch.tensor(C).double()[None, :, :]

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

            input_normalization = kwargs["input_normalization"]
            # perhaps there's a more efficient way to do this?
            # get derivative of neural net output with respect to time

            # this will contain all derivatives of the output with respect to time
            # with shape number of training points x reduced order dimension x 1
            jacobian = torch.permute(self._batch_jacobian(self.ronn, normalized_mu, input_normalization), (1, 0, 2))[:, :, [0]]

            if 'm' in self.operators:
                res1 += torch.matmul(self.operators['m'], jacobian.double())

            initial_condition_loss = torch.mean((pred[self.T0_idx] - self.T0_snapshots.T)**2)
        else:
            initial_condition_loss = 0

        loss1 = torch.mean(torch.sum(res1**2, dim=1)) if type(res1) is not float else res1
        loss2 = torch.mean(torch.sum(res2**2, dim=1)) if type(res2) is not float else res2
        if self.ronn.problem.dirichlet_bc_are_homogeneous:
            boundary_condition_loss = 0
        else:
            boundary_condition_loss = torch.mean((pred[:, 0] - 1.)**2)

        self.value = loss1 + loss2 + initial_condition_loss + self.beta*boundary_condition_loss

        return self.value

    def reinitialize(self, mu):
        normalization = self.normalization
        beta = self.beta
        return PINN_Loss(self.ronn, normalization, beta, mu)


class PDNN_Loss(RONN_Loss_Base):
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

        self.value = torch.mean((pred.T - self.proj_snapshots)**2)

        return self.value

    def reinitialize(self, mu):
        normalization = self.normalization
        return PDNN_Loss(self.ronn, normalization, mu)


class PRNN_Loss(RONN_Loss_Base):
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
        self.value = dict()
        self.value["pdnn_loss"] = self.pdnn_loss(**kwargs)
        self.value["pinn_loss"] = self.pinn_loss(**kwargs)
        self.value["loss"] = self.value["pinn_loss"] + self.omega * self.value["pdnn_loss"]

        return self.value["loss"]

    def reinitialize(self, pdnn_mu, pinn_mu):
        normalization = self.pdnn_loss.normalization
        omega = self.omega
        beta = self.beta
        loss = PRNN_Loss(self.ronn, normalization, omega, beta)
        loss.set_mu(pdnn_mu, pinn_mu)
        return loss




""" Other losses """
class Weighted_PDNN_Loss(RONN_Loss_Base):
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

        self.value = torch.mean(separate_losses)

        return self.value

    def reinitialize(self, mu):
        normalization = self.normalization
        epsilon = self.epsilon
        return Weighted_PDNN_Loss(self.ronn, normalization, mu, epsilon)
