import numpy as np
from numpy.linalg import eig as eig
import cvxpy as cp
from scipy.optimize import fmin, curve_fit
import matplotlib.pyplot as plt

# Calculate sigma' for softplus
def sigma_prime (t, beta):
    return beta / (1.0+np.exp(-beta*t))

# Calculate sigma'' for softplus
def sigma_prime_2 (t, beta):
    return beta * beta * np.exp(beta*t) / ((1.0+np.exp(beta*t))**2)

def sdp_solver(A, dim):
    Y = cp.Variable((dim,dim), PSD=True)
    obj = cp.Maximize(cp.sum(cp.multiply(A,Y)))
    constraints = [cp.diag(Y)<= np.ones(dim)]
    problem = cp.Problem(obj, constraints)
    problem.solve()
    #print("SDP Optimal value: ", problem.value)
    #print("Y: ", Y.value)
    return problem.value

def sdp_feasibility(A, dim, val):
    Y = cp.Variable((dim,dim), PSD=True)
    obj = cp.Maximize(0)
    constraints = [cp.diag(Y) <= np.ones(dim),
                   cp.sum(cp.multiply(A,Y)) >= val]
    problem = cp.Problem(obj, constraints)
    problem.solve()
    return problem.status

def qp_solver(A, dim, delta, c):
    Y = cp.Variable((dim,dim), PSD=True)
    c = np.array([c])
    obj = cp.Maximize(0.5 * cp.sum(cp.multiply(A,Y)) + cp.sqrt(cp.sum(cp.multiply(cp.matmul(c.T,c),Y))))
    constraints = [cp.diag(Y) <= np.ones(dim) * delta * delta]
    problem = cp.Problem(obj, constraints)
    problem.solve(verbose=True)
    print("qp - SDP relax Optimal value: ", problem.value)
    #print("Y*: ", Y.value)
    return np.sqrt(np.sum(np.multiply(np.matmul(c.T,c),Y.value))), 0.5 * np.sum(np.multiply(A,Y.value))

def qp_feasibility(A, dim, delta, c, val):
    Y = cp.Variable((dim,dim), PSD=True)
    c = np.array([c])
    obj = cp.Maximize(0)
    constraints = [cp.diag(Y) <= np.ones(dim) * delta * delta,
                   0.5 * cp.sum(cp.multiply(A,Y)) + cp.sqrt(cp.sum(cp.multiply(cp.matmul(c.T,c),Y))) >= val]
    problem = cp.Problem(obj, constraints)
    problem.solve()
    #print("qp Optimal value: ", problem.value)
    #print("Y*: ", Y.value)
    return problem.status

def cal_dual_opt(psd_M, dim):

    c = cp.Variable(dim,nonneg=True)
    lambda_plus = cp.Variable(nonneg=True)
    obj = cp.Minimize(dim * lambda_plus + cp.sum(c))
    constraints = [lambda_plus >= cp.lambda_max(psd_M - cp.diag(c))]
    problem = cp.Problem(obj, constraints)
    problem.solve(verbose = True)
    print("Dual Optimal value: ", problem.value)
    print("lambda_plus*: ", lambda_plus.value)
    return problem.value

def eq_solver(a, beta):
    # Use the numerical solver to find the root
    
    def myFunc(x):
        return - beta / x * (1.0 / (1.0+np.exp(-beta*(x+a))) - 1.0 / (1.0 + np.exp(-beta*a)))
    
    initial_guess = -a
    y = fmin(myFunc, initial_guess, full_output=True, disp=False)
    return y

"""
A = np.random.rand(200,200)

print(sdp_solver(A,200))
print(cal_dual_opt(A,200))
"""
"""
A = [[2, -1, 0],
     [-1, 2, -1],
     [0, -1, 2]]
b = [1,1,1]

print(qp_solver(A, 3, 0.1, b))

"""
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
#return 1.0/(1.0+np.exp(-b*x))+a*x
    return a*x + b*x**3 + c*x**5
# return a/(x*x+b*np.abs(x)+c) + d/(np.abs(x)+e) + f
    
p0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.05])
    
coeffs, matcov = curve_fit(func, x, y, p0)
    
yaj = func(x, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5])

y_corrected = func(x, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5])

print(coeffs)

#plt.plot(x, y, label='x_opt')
plt.plot(x, y_corrected-y, label='fit_curve')
plt.legend(loc='upper left')
plt.show()

