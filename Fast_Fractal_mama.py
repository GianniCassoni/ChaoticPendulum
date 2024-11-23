import numpy as np
from scipy.integrate import odeint
import multiprocessing
from scipy.linalg import qr

# ODE system
def model(u, t, k, w, f, c):
    x1, x2 = u
    return [x2, f*np.cos(w*t) - k*np.sin(x1) - c*x2]

# Jacobian matrix
def JM(u, c, k):
    return np.array([[0, -1], [-k*np.cos(u), -c]])

def d_JM():
    return np.array([[1, 0], [0, -1]])

def myqr(A):
    Q, R = qr(A)
    nr, nc = 2, 2
    nn = nr
    Q = Q[:, 0:nc]
    R = R[0:nc, :]
    for ii in range(nn):
        if R[ii, ii] < 0:
            Q[:, ii] = -Q[:, ii]
            R[ii, ii:2] = -R[ii, ii:2]
    return Q, R

def LCE(evals, samples_C, G, M, ll_1, ll_2, Qk, Rk, Y0, Yk, R):
    ll_curr = np.zeros(2)
    Qkm = np.zeros((2, 2))
    Ykm = np.zeros((2, 2))
    v_n = evals.T
    lyap = np.zeros((2, grid_size))
    c_k = samples_C
    for H in range(1, grid_size):
        v0 = v_n[0, H]
        A = JM(v0, c_k, k)
        F = np.linalg.lstsq((d_JM() + dt/2.*A), (d_JM() - dt/2.*A), rcond=None)[0]
        Qkm = Qk
        Qk = np.dot(F, Qk)
        Ykm = Yk
        Yk = np.dot(F, Yk)
        Qk, Rk = myqr(Qk)
        ll_curr += np.log(Rk.diagonal())
        lyap[:, H] = ll_curr / t[H]
    ll_1[G, M] = lyap[0, H]
    ll_2[G, M] = lyap[1, H]
    Y0 = np.eye(2)
    Qk, Rk = np.linalg.qr(Y0)
    Yk = Y0
    R = Rk
    return ll_1[G, M], ll_2[G, M], G, M

# Constants
w = 2*np.pi
w_0 = 3/2 * w 
gamma = 1.5
beta = 3/4 * np.pi
c = 2 * beta
k = w_0**2
f = gamma * w_0**2
y0 = y1 = 0
init_cond = y0, y1
t_max = 10
dt = 0.01
grid_size = int(t_max / dt) + 1
t = np.arange(0, t_max + dt, dt)
atol = 1e-10
rtol = 1e-10

# Pseudo Spectral Projection
dx1 = 0.001
C_fin = 8
C_in = 1
grid_C = int(C_fin / dx1) - int(C_in / dx1) + 1
samples_C = np.linspace(C_in, C_fin, grid_C)
dx2 = 0.1
K_fin = 100
K_in = 40
grid_K = int(K_fin / dx2) - int(K_in / dx2) + 1
samples_K = np.linspace(K_in, K_fin, grid_K)

# Initialization
ll_1 = np.zeros((grid_C, grid_K))
ll_2 = np.zeros((grid_C, grid_K))
v0 = np.ones(2)
Qk = np.eye(2)
Rk = np.eye(2)
Y0 = np.eye(2)
Yk = np.eye(2)
R = np.eye(2)

pool = multiprocessing.Pool(processes=18)

def process_M(M):
    f = gamma * samples_K[M]
    evals = [odeint(model, init_cond, t, args=(samples_K[M], w, f, sample)) for sample in samples_C.T]
    results = pool.starmap(LCE, [(evals[G], samples_C[G], G, M, ll_1, ll_2, Qk, Rk, Y0, Yk, R) for G in range(grid_C)])
    for result in results:
        ll_1[result[2], result[3]] = result[0]
        ll_2[result[2], result[3]] = result[1]
    print(M)

for M in range(grid_K):
    process_M(M)

pool.close()

np.savetxt('ll_1.txt', ll_1, fmt='%.2f')
np.savetxt('ll_2.txt', ll_2, fmt='%.2f')
