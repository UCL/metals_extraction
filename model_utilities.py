# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:36:01 2025

@author: Arun
"""

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torchdiffeq import odeint


def fermentation(t, x, u, theta):
    '''

    Parameters
    ----------
    t : array
        A sequence of time points for which to solve for x. The initial value 
        point should be the first element of this sequence.
    x : array
        State variables.
    u : array
        Process inputs or controls.
    theta : array
        Model parameters.

    Returns
    -------
    Callable process model function.

    '''
    
    r = ((theta[0] * x[1])/(theta[1] + x[1]))
    dx1dt = (r - u[0] - theta[3]) * x[0]
    dx2dt = ((-r * x[0])/theta[2]) + (u[0] * (u[1] - x[1]))
    return [dx1dt, dx2dt]

def dynamicsimulation(initx, u, theta, T, N):
    '''

    Parameters
    ----------
    initx : array
        Initial condition on x.
    u : array
        Process inputs or controls.
    theta : array
        Model parameters.
    T : array
        Array with discretized time points.
    N : int
        Number of discrete control intervals.

    Returns
    -------
    Array of state variables solved for specified time points.

    '''
    
    x0 = []
    x0 += [initx]
    y_hat = []
    for i in range(N):
        t = np.linspace(T[i], T[i+1], 100)
        y_hat += [solve_ivp(fermentation, (0,300), x0[-1], method='BDF', t_eval = t, rtol = 1e-5, atol = 1e-8, args=(u[:,i], theta)).y.T]
        x0 += [y_hat[-1][-1]]
    return np.array(x0)

def torchsimulator(x0, u, t, theta):
    '''

    Parameters
    ----------
    x0 : tensor
        Initial condition on x (state variables).
    u : tensor
        Process inputs or controls.
    t : tensor
        Time points for which to solve for x.
    theta : tensor
        Model parameters.

    Returns
    -------
    Tensor of state variables, solved for specified time points.

    '''
    
    def rhs(t, x):
        r = ((theta[0] * x[1]) / (theta[1] + x[1]))
        dx1dt = (r - u[0] - theta[3]) * x[0]
        dx2dt = ((-r * x[0]) / theta[2]) + (u[0] * (u[1] - x[1]))
        return torch.stack([dx1dt, dx2dt])
    
    return odeint(rhs, x0, t)

def adjsens(x0, u, t, theta):
    '''

    Parameters
    ----------
    x0 : tensor
        Initial condition on x (state variables).
    u : tensor
        Process inputs or controls.
    t : tensor
        Time points for which to solve for x.
    theta : tensor
        Model parameters.

    Returns
    -------
    Array of sensitivity matrix at specified time points.

    '''
    
    def rhs(t, x):
        r = ((theta[0] * x[1]) / (theta[1] + x[1]))
        dx1dt = (r - u[0] - theta[3]) * x[0]
        dx2dt = ((-r * x[0]) / theta[2]) + (u[0] * (u[1] - x[1]))
        return torch.stack([dx1dt, dx2dt])
    
    x_pred = odeint(rhs, x0, t)
    
    # Compute sensitivities dynamically at each time step
    sensitivity_x1 = []
    sensitivity_x2 = []

    for i in range(len(t)):
        x_t = x_pred[i]
        
        # Sensitivity of x(t) w.r.t. theta
        grad_x1 = torch.autograd.grad(x_t[0], theta, retain_graph=True, allow_unused=True)[0]
        grad_x2 = torch.autograd.grad(x_t[1], theta, retain_graph=True, allow_unused=True)[0]
        
        sensitivity_x1.append(grad_x1.detach().numpy())
        sensitivity_x2.append(grad_x2.detach().numpy())

    # Convert to arrays for plotting or analysis
    sensitivity_x1 = np.array(sensitivity_x1)  # shape: [T, P]
    sensitivity_x2 = np.array(sensitivity_x2)  # shape: [T, P]
    adj_sens = np.stack((sensitivity_x1, sensitivity_x2))
    adj_rev_sens = np.transpose(adj_sens,(1,0,2))
    
    return adj_rev_sens

def dirsens(xs0, u, t, theta):
    '''

    Parameters
    ----------
    xs0 : tensor
        Initial condition on x (state variables) and s (sensitivity terms).
        Sensitivity terms are the partial derivatives of x w.r.t theta (model
        parameters).
    u : tensor
        Process inputs or controls.
    t : tensor
        Time points for which to solve for x.
    theta : tensor
        Model parameters.

    Returns
    -------
    Array of sensitivity matrix at specified time points.

    '''
    
    n_x = 2
    n_theta = 4
    
    def oderhs(x, u, theta):
        r = ((theta[0] * x[1]) / (theta[1] + x[1]))
        dx1dt = (r - u[0] - theta[3]) * x[0]
        dx2dt = ((-r * x[0]) / theta[2]) + (u[0] * (u[1] - x[1]))
        return torch.stack([dx1dt, dx2dt])
    
    def sensrhs(t, xs):
        x = xs[:n_x]
        s = xs[n_x:].reshape(n_x, n_theta)
        
        r = ((theta[0] * x[1]) / (theta[1] + x[1]))
        dx1dt = (r - u[0] - theta[3]) * x[0]
        dx2dt = ((-r * x[0]) / theta[2]) + (u[0] * (u[1] - x[1]))
        dxdt = torch.stack([dx1dt, dx2dt])
        
        # Jacobians
        Jx = torch.autograd.functional.jacobian(lambda x_: oderhs(x_, u, theta), x, create_graph=True)
        Jtheta = torch.autograd.functional.jacobian(lambda theta_: oderhs(x, u, theta_), theta, create_graph=True)

        # Sensitivity RHS: dS/dt = Jx @ S + Jtheta
        dSdt = Jx @ s + Jtheta  # shape: [n_state, n_param]
        dSdt_flat = dSdt.reshape(-1)
        
        return torch.cat([dxdt, dSdt_flat])
    
    return odeint(sensrhs, xs0, t)[:, 2:].reshape(-1, 2, 4).detach().numpy()

def hessFIM(x0, u, t, theta):
    '''

    Parameters
    ----------
    x0 : tensor
        Initial condition on x (state variables).
    u : tensor
        Process inputs or controls.
    t : tensor
        Time points for which x is solved for.
    theta : tensor
        Model parameters.

    Returns
    -------
    Tensor of FIM matrix derived directly from log-likelihood.
    FIM derived as the negative hessian of log-likelihood function.

    '''
    
    n_x = 2
    
    def llf(x0, u, theta):
        def oderhs(t, x):
            r = ((theta[0] * x[1]) / (theta[1] + x[1]))
            dx1dt = (r - u[0] - theta[3]) * x[0]
            dx2dt = ((-r * x[0]) / theta[2]) + (u[0] * (u[1] - x[1]))
            return torch.stack([dx1dt, dx2dt])
        
        x_pred = odeint(oderhs, x0, t)
        
        chisq = torch.sum(x_pred.sum(axis=0)*(1/(torch.tensor([0.01,0.05]))))
        llfobj = -(((x_pred.shape[0] * n_x) / 2) * torch.log(torch.tensor(2 * torch.pi)) + (x_pred.shape[0] / 2) * (2 * torch.sum(torch.log(torch.diag(torch.linalg.cholesky(torch.tensor([[0.01,0.0],[0.0,0.05]])))))) + (chisq / 2))
        
        return llfobj
        
    FIM_hess = -torch.autograd.functional.hessian(lambda theta_: llf(x0, u, theta_), theta, create_graph=True)
    
    return FIM_hess
    