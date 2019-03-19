import numpy as np
from numpy.linalg import eig as eig
import cvxpy as cp
from scipy.optimize import fmin, curve_fit
import matplotlib.pyplot as plt

def qp_solver(A, dim, delta, c):
    Y = cp.Variable((dim + 1, dim + 1), PSD=True)
    c = np.array([c])
    A1 = np.concatenate((np.multiply(0.5, A), c.T), axis = 1)
    A1 = np.concatenate((A1, np.zeros((1, dim + 1))), axis = 0)
    obj = cp.Maximize(cp.sum(cp.multiply(A1, Y)))
    constraints = [cp.diag(Y) <= np.append(np.ones(dim) * delta * delta, [1]),
                   cp.abs(Y[dim, 0:dim]) <= delta]
    problem = cp.Problem(obj, constraints)
    problem.solve("SCS", gpu=False, verbose=False)
    # print("qp - SDP relax Optimal value: ", problem.value)
    fo_term = np.sum(np.multiply(c, Y.value[dim, 0:dim]))
    return fo_term, problem.value - fo_term

def qp_feasibility(A, dim, delta, c, val):
    Y = cp.Variable((dim + 1, dim + 1), PSD=True)
    c = np.array([c])
    A1 = np.concatenate((np.multiply(0.5, A), c.T), axis = 1)
    A1 = np.concatenate((A1, np.zeros((1, dim + 1))), axis = 0)
    obj = cp.Maximize(0)
    constraints = [cp.diag(Y) <= np.append(np.ones(dim) * delta * delta, [1]),
                   cp.abs(Y[dim, 0:dim]) <= delta,
                   cp.sum(cp.multiply(A1, Y)) >= val]
    problem = cp.Problem(obj, constraints)
    problem.solve("SCS", gpu=False)
    return problem.status

def eq_solver(a, beta):
    # Use the numerical solver to find the root
    
    def myFunc(x):
        return - beta / x * (1.0 / (1.0+np.exp(-beta*(x+a))) - 1.0 / (1.0 + np.exp(-beta*a)))
    
    initial_guess = -a
    y = fmin(myFunc, initial_guess, full_output=True, disp=False)
    return y

"""
Below are toy examples to test sdp solvers above
"""
"""
A = np.random.rand(200,200)

A = [[2, -1, 0],
     [-1, 2, -1],
     [0, -1, 2]]
b = [1,1,1]

print(qp_feasibility(A, 3, 0.1, b, 0.2))

N = 1000
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)
f_exp = np.zeros(N)
beta = 5.0
for i in range(N):
    x[i] = -10.0 + 20.0 / N * i
    temp = eq_solver(x[i], beta)
    y[i] = temp[0][0] # x_opt

def func(x, a, b, c, d, e, f):
    return 1.0/(1.0+np.exp(-b*x))+a*x
    return a*x + b*x**3 + c*x**5
    return a/(x*x+b*np.abs(x)+c) + d/(np.abs(x)+e) + f
    
p0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.05])
    
coeffs, matcov = curve_fit(func, x, y, p0)
    
yaj = func(x, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5])

y_corrected = func(x, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5])

print(coeffs)

plt.plot(x, y, label='x_opt')
plt.plot(x, y_corrected-y, label='fit_curve')
plt.legend(loc='upper left')
plt.show()
"""
