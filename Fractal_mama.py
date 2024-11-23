###############################################################################
import numpy as np
from scipy.integrate import odeint
import multiprocessing
from scipy.linalg import qr
###################################à Lyapunov

#ODE system
def model(u, t, k, w, f,c):
	x1, x2 = u
	f_vect = [x2, f*np.cos(w*t) - k*np.sin(x1) - c*x2]
	return f_vect
#Jacobian matrix
def JM(u,c,k):
 	return np.array([[0,-1],[-k*np.cos(u),-c]])
def d_JM():
 	return np.array([[1,0],[0,-1]])
def myqr(A):
    Q, R = qr(A)
    nr=2
    nc=2
    nn = nr
    Q = Q[:, 0:nc]
    R = R[0:nc, :]
    for ii in range(nn):
        if R[ii, ii] < 0 :
            Q[:, ii] = -Q[:, ii]
            R[ii, ii:2] = -R[ii, ii:2]

    return Q, R

def LCE(evals, samples_C, G,M, ll_1,ll_2,v0,Qk,Rk,Qkm,Ykm,Y0,Yk,R):
    #for G in range(0,grid_C):
    ll_curr = np.zeros([1, 2]);
    Qkm=np.zeros([2,2])
    Temp=np.zeros([2,2])
    Ykm=np.zeros([2,2])
    tmp=np.zeros([2,2])
    v_n = evals.T #transpose the solution
    lyap=np.zeros([2,grid_size])
    c_k=samples_C
    for H in range(1, grid_size ):
        v0 = v_n[0,H]
        A=JM(v0,c_k,k)
        F= np.linalg.lstsq((d_JM()+dt/2.*A),(d_JM()-dt/2.*A),rcond=None)[0]
        Qkm=Qk
        Qk=np.dot(F,Qk)
        Ykm=Yk
        Yk=np.dot(F,Yk)
        Qk, Rk = myqr(Qk)
        ll_curr=ll_curr+np.log(Rk.diagonal().T)
        lyap[:,H]=ll_curr/t[H]
    ll_1[G,M]=lyap[0,H]
    ll_2[G,M]=lyap[1,H]
    Y0 = np.eye(2);
    Qk, Rk= np.linalg.qr(Y0)
    Yk = Y0
    R = Rk
    return ll_1[G,M], ll_2[G,M], G, M
    
##################################################### C uniform
w =2*np.pi
w_0 =3/2*w 
gamma =1.5
beta =3/4*np.pi
c=2*beta
k=w_0*w_0
f=gamma*(w_0)*(w_0)
y0  = 0
y1  = 0
init_cond   = y0, y1
t_max       = 10
dt          = 0.01
grid_size   = int(t_max/dt) + 1
t           = np.array([i*dt for i in range(grid_size)])
t_interest  = len(t)/2
atol = 1e-10
rtol = 1e-10
# w is no longer deterministic
# c_left= 2*beta-1
# c_right= 2*beta+1
# create uniform distribution object
# unif_distr = cp.J(cp.Uniform(w_left,w_right), cp.Uniform(w_left,w_right))
# unif_distr =cp.Normal(2*beta, 0.1)
# quadrature_order
########################  Pseudo Spectral Projection 
#tratta il problema come una black box
# node and weights
dx1=0.035
C_fin=8
C_in=1
grid_C  = int(C_fin/dx1)  - int(C_in/dx1) +1
samples_C=np.linspace(C_in, C_fin, grid_C)
dx2=0.3
K_fin=100
K_in=40
grid_K  = int(K_fin/dx2)  - int(K_in/dx2) +1
samples_K=np.linspace(K_in, K_fin, grid_K)

# Generate samples for both schemes
 #empt

# samples = np.random.normal(2*beta, 0.1, number_of_samples)
evaluations = []
#evaluations = {}
evaluations_ss = {}
ll_1= np.zeros([grid_C ,grid_K])
ll_2= np.zeros([grid_C ,grid_K])
v0 = np.ones(2) #initial condition
Qk=np.zeros([2,2])
Rk=np.zeros([2,2])
Qkm=np.zeros([2,2])
Ykm=np.zeros([2,2])
Y0 = np.eye(2);
Qk, Rk= np.linalg.qr(Y0)
Yk = Y0
R = Rk

pool = multiprocessing.Pool(processes=18)

for M in range(0,grid_K):
    f=gamma*samples_K[M]
    evals = [odeint(model, init_cond, t, args=(samples_K[M], w, f,sample)) for sample in samples_C.T]
    results = pool.starmap(LCE, ([evals[G], samples_C[G],G,M,ll_1,ll_2,v0,Qk,Rk,Qkm,Ykm,Y0,Yk,R] for G in range(0,grid_C)), chunksize=1)
    for i in range(0, grid_C):
        ll_1[results[i][2],results[i][3]] = results[i][0]
        ll_2[results[i][2],results[i][3]] = results[i][1]
    print(M)

pool.close()

np.savetxt('ll_1.txt',ll_1,fmt='%.2f')
np.savetxt('ll_2.txt',ll_2,fmt='%.2f') 
#fig, ax = plt.subplots(1, 1)
#
## plots filled contour plot
#cntr=ax.contourf(samples_K,samples_C,ll_2)
## DATA=plt.imshow( ll_1,interpolation='gaussian')
#ax.set_title('$\\lambda_1$')
#ax.set_xlabel('K')
#ax.set_ylabel('C')
#plt.show()
#cbar = fig.colorbar(cntr,ax=ax)
#cbar.set_ticks(np.arange(-6,6,0.5))
#plt.show()
## Sample mean and variance
#
## smean = np.mean(evaluations, axis=0)
## svar = np.std(evaluations, axis=0)
##contour = plt.contour(samples_K, samples_C, ll_1)
##plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
##c = ('#ff0000', '#ffff00', '#0000FF', '0.6', 'c', 'm')
##contour_filled = plt.contourf(samples_K, samples_C, ll_1, colors=c)
##plt.colorbar(contour_filled)
##ax.set_title('$\\lambda_1$')
##ax.set_xlabel('K')
##ax.set_ylabel('C')
##plt.show()
##cbar.set_ticks(np.arange(-3.5,3,0.5))
#plt.show()
#plt.close()
################################àà
##plt.subplot(131)
##plt.imshow(ll_1,interpolation='nearest')
#
##plt.subplot(132)
##plt.imshow(ll_1)
#
##plt.subplot(133)
#fig, ax = plt.subplots(figsize=(10,10))
#ax.imshow(ll_2, cmap=plt.cm.RdYlGn_r,interpolation='gaussian',origin='lower',extent=[40,100,1,8])
##ax.imshow(ll_1,origin='lower',extent=[0,200,0,20])
#ax.set_aspect(10)
#cbar = fig.colorbar(cntr,ax=ax)
#cbar.set_ticks(np.arange(-20,3,0.5))
#
#
#ax.set_title('$\\lambda_2$')
#ax.set_xlabel('K') 
#ax.set_ylabel('C')
#plt.show()