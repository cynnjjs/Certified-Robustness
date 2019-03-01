import cvxpy as cp
import numpy as np

Y = cp.Variable((3,3), PSD=True)
A = [[2, -1, 0],
           [-1, 2, -1],
           [0, -1, 2]]
b = [1,1,1]

obj = cp.Maximize(cp.sum(cp.multiply(A,Y)))
constraints = [cp.diag(Y)<= b]
problem = cp.Problem(obj, constraints)
problem.solve()
print("Optimal value: ", problem.value)
print("Y: ", Y.value)
