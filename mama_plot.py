import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
# Load the data from the file
ll_1 = np.loadtxt('ll_1_good.txt')
ll_2 = np.loadtxt('ll_2_good.txt')
# Now 'data' contains the contents of the file 'text.txt'
fig, ax = plt.subplots(1, 1)
#
## plots filled contour plot
#cntr=ax.contourf(samples_K,samples_C,ll_2)
DATA=plt.imshow( ll_1,interpolation='gaussian')
ax.set_title('$\\lambda_1$')
ax.set_xlabel('K')
ax.set_ylabel('C')
plt.show()
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
fig, ax = plt.subplots(figsize=(10,10))
#ax.imshow(ll_2, cmap='Spectral_r',interpolation='gaussian',origin='lower',extent=[40,100,1,8])
ax.imshow(ll_1,cmap='inferno_r',interpolation='hermite', origin='lower',extent=[0,200,0,20])
ax.set_aspect(10)
#cbar = fig.colorbar(cntr,ax=ax)
#cbar.set_ticks(np.arange(-20,3,0.5))
#
#

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.tight_layout()
plt.grid(color='gray', linestyle='dashed',linewidth=1.5, alpha = 0.2)
ax.set_title('$\\lambda_2$')
ax.set_xlabel('K') 
ax.set_ylabel('C')
plt.show()