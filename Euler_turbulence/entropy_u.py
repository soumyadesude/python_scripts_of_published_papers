import numpy as np
#import pyfftw.interfaces.numpy_fft as fp
import time 
from astropy.table import Table 
import random
import h5py
import cupy as cp


dev1 = cp.cuda.Device(0)
dev1.use()

cp._core.set_routine_accelerators(['cub', 'cutensor'])

def Get_kx(lx):
    if lx <= Nx/2:
      return lx
    else:
      return lx-Nx

def Multiplicity_factor(lx,ly):
    kx = Get_kx(lx)
    
    if ly>0:
      factor = 0.5
    else:
      factor = 0.5
    
    if (kx==Nx/2):
      return 2*factor   # for kx = -Nx/2  ;  Ignoring corner which would have factor 4
    else:
      return factor

Nx = 2048
Nz = 2048

i = 0

count = 1

entropy = []
time = []
while (i <= 50):
   with h5py.File("Soln_%f.h5"%(i),'r') as f:
       V1 = cp.asarray(f['Vkx'][()])
       V3 = cp.asarray(f['Vkz'][()])

       B1 = cp.asarray(f['Bkx'][()])
       B3 = cp.asarray(f['Bkz'][()])

   #print("hi")

   totE = cp.sum(cp.abs(V1)**2 + cp.abs(V3)**2+cp.abs(B1)**2 + cp.abs(B3)**2)
   totE -= cp.sum(cp.abs(V1[:,0])**2+cp.abs(V3[:,0])**2+cp.abs(B1[:,0])**2+cp.abs(B3[:,0])**2)/2

   local_entropy = 0
   local_entropy_1d = 0
   local_entropy_tot = 0

   #for lx in range(0, Nx):
   #  for ly in range(1, Nz//2+1):
   #modal_energy = Multiplicity_factor(lx,ly)*(cp.abs(V1[lx, ly])**2 + cp.abs(V3[lx, ly])**2+cp.abs(B1[lx, ly])**2 + cp.abs(B3[lx, ly])**2) 
   modal_energy = 0.5*(cp.abs(V1[0:Nx, 1:Nz//2+1])**2 + cp.abs(V3[0:Nx, 1:Nz//2+1])**2+cp.abs(B1[0:Nx, 1:Nz//2+1])**2 + cp.abs(B3[0:Nx, 1:Nz//2+1])**2) 

   prob = modal_energy/ totE

         #if (prob > 1e-15):
   local_entropy = 2*cp.nansum((-prob * cp.log(prob))/cp.log(2))
         #print (local_entropy)

   #for lx in range(0, Nx//2):
   #      modal_energy = Multiplicity_factor(lx,0)*(cp.abs(V1[lx, 0])**2 + cp.abs(V3[lx, 0])**2+cp.abs(B1[lx, 0])**2 + cp.abs(B3[lx, 0])**2) 
   modal_energy = 0.5*(cp.abs(V1[1:Nx//2, 0])**2 + cp.abs(V3[1:Nx//2, 0])**2+cp.abs(B1[1:Nx//2, 0])**2 + cp.abs(B3[1:Nx//2, 0])**2) 
  

   prob = modal_energy/ totE

         #if (prob > 1e-15):
              #local_entropy_1d += ((-prob * cp.log(prob))/cp.log(2))
             #else:
   local_entropy_1d = 2*cp.nansum((-prob * cp.log(prob))/cp.log(2))


   modal_energy = 0.5*(cp.abs(V1[0, 0])**2 + cp.abs(V3[0, 0])**2+cp.abs(B1[0, 0])**2 + cp.abs(B3[0, 0])**2)

   local_entropy_1d += modal_energy/ totE
   
         #print (local_entropy)
   
   local_entropy_tot = local_entropy+local_entropy_1d

   entropy.append(cp.asnumpy(local_entropy_tot))
   time.append(cp.asnumpy(i))
   print (i, local_entropy_tot)
   
   i = i+0.1

print ("done")


d = Table([time, entropy])
np.savetxt("entropy_E.txt", d)

#print (local_entropy)


