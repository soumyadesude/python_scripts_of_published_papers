import numpy as np
import scipy as sp
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
plt.rcParams['xtick.major.size'] = 0.7*4.0
plt.rcParams['xtick.major.width'] = 2*0.5
plt.rcParams['xtick.minor.size'] = 0.7*2.5
plt.rcParams['xtick.minor.width'] = 2*0.5
plt.rcParams['ytick.major.size'] = 0.7*4.0
plt.rcParams['ytick.major.width'] = 2*0.5
plt.rcParams['ytick.minor.size'] = 0.7*2.5
plt.rcParams['ytick.minor.width'] = 2*0.5
A=1.75*9.3#1.5*9.3
font = {'family' : 'serif', 'weight' : 'normal', 'size' : A}
plt.rc('font', **font)

fig, axes = plt.subplots(nrows=2, ncols=3,figsize = (7, 9), gridspec_kw={'wspace':0.15, 'hspace':-0.6, 'width_ratios': [1, 1, 1.106], 'height_ratios': [1, 1]})

#### Case 1

##### # 2d TGV fields #######

def hdf5_reader_2d(filename,dataset):
    file_V1_read = h5py.File(filename)
    dataset_V1_read = file_V1_read["/"+dataset]
    V1=dataset_V1_read[:,:]

    return V1


def inverse_transform(Ak):
    A = (np.fft.irfftn(Ak)*(np.shape(Ak)[0]*np.shape(Ak)[1]*(2*np.shape(Ak)[2]-2)))
    return A


V1_k = hdf5_reader_2d("2d_random_tgv_k11_pdealias_chaos/Soln_0.000000.h5", "Vkx")
V3_k = hdf5_reader_2d("2d_random_tgv_k11_pdealias_chaos/Soln_0.000000.h5", "Vkz")

Nx = 512
Nz = 512


kx = np.linspace(0,Nx-1, Nx)
kx[Nx//2+1:Nx] = kx[Nx//2+1:Nx]-Nx

kz = np.linspace(0, Nz//2, Nz//2+1)


kx_mesh, kz_mesh = np.meshgrid(kx, kz,indexing = 'ij')

omega_z = 1j*(kx_mesh*V3_k- kz_mesh*V1_k)

V1 = np.fft.irfft2(V1_k)*(Nx*Nz)
V2 = np.fft.irfft2(V3_k)*(Nx*Nz)

omega_z_r = np.fft.irfft2(omega_z)*(Nx*Nz)

#l_lim = np.min(omega_z_r)
#u_lim = np.max(omega_z_r)

l_lim = np.min(V2)
u_lim = np.max(V2)


#print(l_lim, u_lim, np.sum(np.transpose(omega_z_r)))

#print (np.max(omega_z_r), np.min(omega_z_r))

x = np.linspace(0,2*np.pi, Nx)#, endpoint=False)
z = np.linspace(0,2*np.pi, Nz)#, endpoint=False)

#x_mesh, z_mesh = np.meshgrid(x, z,indexing = 'ij')

#fig, axes = plt.subplots()
g=7

#density = axes[0, 0].pcolor(x, z, np.transpose(omega_z_r), cmap='jet')#, shading='auto')#,vmin=-50, vmax=50)

density = axes[0, 0].pcolor(x, z, np.transpose(omega_z_r)*(5/24), cmap='jet', vmin=-5, vmax=5, shading='auto')#,vmin=-50, vmax=50)

axes[0, 0].quiver(x[::g], z[::g], np.transpose(V1[::g, ::g]), np.transpose(V2[::g, ::g]), units='width')#, width=0.0065)

cb1 = fig.colorbar(density, fraction=0.045, ax=axes[0, 0])
cb1.ax.tick_params(labelsize=A)

cb1.remove()
#cb1.set_ticks([-24, -12, 0, 12, 24])

#cb1.set_label('$\omega_z$', rotation=90, labelpad=10)
#cb1.set_label(r'$\omega$', rotation=0, labelpad=10)

#axes[0, 0].set_xlabel('$x$', fontsize=A)
axes[0, 0].set_ylabel('$y$', fontsize=A)
axes[0, 0].set_title('   Run A \n (a): $t=0$', fontsize=A, pad=5)
axes[0, 0].set_aspect(1)
axes[0, 0].set_ylim(0, 2*np.pi)
axes[0, 0].set_xlim(0, 2*np.pi)

axes[0, 0].set_yticks([0, 2*np.pi])
axes[0, 0].set_xticks([0, 2*np.pi])
axes[0, 0].axes.xaxis.set_visible(False)

label = ["$0$", "$2\pi$"]

axes[0, 0].set_xticklabels(label, fontsize=A)
axes[0, 0].set_yticklabels(label, fontsize=A)





###

V1_k = hdf5_reader_2d("2d_random_tgv_k11_pdealias_chaos/Soln_999.990000.h5", "Vkx")
V3_k = hdf5_reader_2d("2d_random_tgv_k11_pdealias_chaos/Soln_999.990000.h5", "Vkz")

Nx = 512
Nz = 512


kx = np.linspace(0,Nx-1, Nx)
kx[Nx//2+1:Nx] = kx[Nx//2+1:Nx]-Nx

kz = np.linspace(0, Nz//2, Nz//2+1)


kx_mesh, kz_mesh = np.meshgrid(kx, kz,indexing = 'ij')

omega_z = 1j*(kx_mesh*V3_k- kz_mesh*V1_k)

V1 = np.fft.irfft2(V1_k)*(Nx*Nz)
V2 = np.fft.irfft2(V3_k)*(Nx*Nz)

omega_z_r = np.fft.irfft2(omega_z)*(Nx*Nz)

#l_lim = np.min(omega_z_r)
#u_lim = np.max(omega_z_r)

l_lim = np.min(V2)
u_lim = np.max(V2)


#print(l_lim, u_lim, np.sum(np.transpose(omega_z_r)))

#print (np.max(omega_z_r), np.min(omega_z_r))

x = np.linspace(0,2*np.pi, Nx)#, endpoint=False)
z = np.linspace(0,2*np.pi, Nz)#, endpoint=False)

#x_mesh, z_mesh = np.meshgrid(x, z,indexing = 'ij')

#fig, axes = plt.subplots()
g=26

density = axes[1, 0].pcolor(x, z, np.transpose(omega_z_r)*(12/28), cmap='jet', vmin=-12, vmax=12, shading='auto') #,vmin=-28, vmax=28) #, shading='auto',vmin=-28, vmax=28)
axes[1, 0].quiver(x[::g], z[::g], np.transpose(V1[::g, ::g]), np.transpose(V2[::g, ::g]), units='width', scale=3, scale_units='xy', angles='xy', width=0.0065)

cb1 = fig.colorbar(density, fraction=0.045, ax=axes[1, 0])
cb1.ax.tick_params(labelsize=A)



#cb1.set_ticks([-24, -12, 0, 12, 24])

cb1.remove()
#cb1.set_label('$\omega_z$', rotation=90, labelpad=10)

axes[1, 0].set_xlabel('$x$', fontsize=A)
axes[1, 0].set_ylabel('$y$', fontsize=A)
#axes[0, 2].set_title(r'$(c)$ $t=120$', fontsize=A, pad=15)

axes[1, 0].set_title('(b): $t=1000$', fontsize=A, pad=5)

axes[1, 0].set_ylim(0, 2*np.pi)
axes[1, 0].set_xlim(0, 2*np.pi)
axes[1, 0].set_aspect(1)

axes[1, 0].set_yticks([0, 2*np.pi])
axes[1, 0].set_xticks([0, 2*np.pi])
axes[1, 0].axes.yaxis.set_visible(True)

axes[1, 0].axes.xaxis.set_visible(True)

label = ["$0$", "$2\pi$"]

axes[1, 0].set_xticklabels(label, fontsize=A)
axes[1, 0].set_yticklabels(label, fontsize=A)



#### Case 1 end

#### Case 2

V1_k = hdf5_reader_2d("mode0/Soln_0.000000.h5", "Vkx")
V3_k = hdf5_reader_2d("mode0/Soln_0.000000.h5", "Vkz")

Nx = 512
Nz = 512


kx = np.linspace(0,Nx-1, Nx)
kx[Nx//2+1:Nx] = kx[Nx//2+1:Nx]-Nx

kz = np.linspace(0, Nz//2, Nz//2+1)


kx_mesh, kz_mesh = np.meshgrid(kx, kz,indexing = 'ij')

omega_z = 1j*(kx_mesh*V3_k- kz_mesh*V1_k)

V1 = np.fft.irfft2(V1_k)*(Nx*Nz)
V2 = np.fft.irfft2(V3_k)*(Nx*Nz)

omega_z_r = np.fft.irfft2(omega_z)*(Nx*Nz)

#l_lim = np.min(omega_z_r)
#u_lim = np.max(omega_z_r)

l_lim = np.min(V2)
u_lim = np.max(V2)


#print(l_lim, u_lim, np.sum(np.transpose(omega_z_r)))

#print (np.max(omega_z_r), np.min(omega_z_r))

x = np.linspace(0,2*np.pi, Nx)#, endpoint=False)
z = np.linspace(0,2*np.pi, Nz)#, endpoint=False)

#x_mesh, z_mesh = np.meshgrid(x, z,indexing = 'ij')

#fig, axes = plt.subplots()
g=22

density = axes[0, 1].pcolor(x, z, np.transpose(omega_z_r), cmap='jet', vmin=-5, vmax=5, shading='auto')#,vmin=-, vmax=7)
axes[0, 1].quiver(x[::g], z[::g], np.transpose(V1[::g, ::g]), np.transpose(V2[::g, ::g]), units='width', scale=14, scale_units='xy', angles='xy', width=0.0058)

cb1 = fig.colorbar(density, fraction=0.045, ax=axes[0, 1])
#cb1.ax.tick_params(labelsize=A)

#cb1.set_ticks([-5, -3, 0, 3, 5])
#cb1.set_ticks([-5, 0, 5])
#cb1.set_ticklabels(["$-5$", "$0$", "$5$"])
cb1.ax.tick_params(labelsize=A)
cb1.remove()
#cb1.set_label(r'$\omega$', rotation=0, labelpad=10)

#cb1.set_label('$\omega_z$', rotation=0, labelpad=10)

#axes[0, 2].set_xlabel('$x$', fontsize=A)
#axes[0, 2].set_ylabel('$y$', fontsize=A)
axes[0, 1].set_title('   Run B \n (c): $t=0$', fontsize=A, pad=5)
axes[0, 1].set_aspect(1)
axes[0, 1].set_ylim(0, 2*np.pi)
axes[0, 1].set_xlim(0, 2*np.pi)

axes[0, 1].set_yticks([0, 2*np.pi])
axes[0, 1].set_xticks([0, 2*np.pi])
axes[0, 1].axes.xaxis.set_visible(False)


axes[0, 1].axes.yaxis.set_visible(False)

label = ["$0$", "$2\pi$"]

axes[0, 1].set_xticklabels(label, fontsize=A)
axes[0, 1].set_yticklabels(label, fontsize=A)


##
V1_k = hdf5_reader_2d("mode0/Soln_999.990000.h5", "Vkx")
V3_k = hdf5_reader_2d("mode0/Soln_999.990000.h5", "Vkz")

Nx = 512
Nz = 512


kx = np.linspace(0,Nx-1, Nx)
kx[Nx//2+1:Nx] = kx[Nx//2+1:Nx]-Nx

kz = np.linspace(0, Nz//2, Nz//2+1)


kx_mesh, kz_mesh = np.meshgrid(kx, kz,indexing = 'ij')

omega_z = 1j*(kx_mesh*V3_k- kz_mesh*V1_k)

V1 = np.fft.irfft2(V1_k)*(Nx*Nz)
V2 = np.fft.irfft2(V3_k)*(Nx*Nz)

omega_z_r = np.fft.irfft2(omega_z)*(Nx*Nz)

#l_lim = np.min(omega_z_r)
#u_lim = np.max(omega_z_r)

l_lim = np.min(V2)
u_lim = np.max(V2)


#print(l_lim, u_lim, np.sum(np.transpose(omega_z_r)))

#print (np.max(omega_z_r), np.min(omega_z_r))

x = np.linspace(0,2*np.pi, Nx)#, endpoint=False)
z = np.linspace(0,2*np.pi, Nz)#, endpoint=False)

x_mesh, z_mesh = np.meshgrid(x, z,indexing = 'ij')

#fig, axes = plt.subplots()
g=30#25

density = axes[1, 1].pcolor(x_mesh, z_mesh, omega_z_r, cmap='jet',vmin=-12, vmax=12, shading='auto')#,vmin=-28, vmax=28)
axes[1, 1].quiver(x[::g], z[::g], np.transpose(V1[::g, ::g]), np.transpose(V2[::g, ::g]), units='width', scale=13, scale_units='xy', angles='xy', width=0.0058)

cb1 = fig.colorbar(density, fraction=0.045, ax=axes[1, 1])
cb1.ax.tick_params(labelsize=A)

#cb1.set_ticks([-12, -6, 0, 6, 12])
#cb1.set_ticks([-12, -6, 0, 6, 12])

cb1.set_ticks([-12, 0, 12])
cb1.set_ticklabels(["$-12$", "$0$", "$12$"])
cb1.ax.tick_params(labelsize=A)
cb1.remove()

#cb1.set_label('$\omega_z$', rotation=0, labelpad=10)

axes[1, 1].set_xlabel('$x$', fontsize=A, labelpad=5.5)
#axes[1, 2].set_ylabel('$y$', fontsize=A)
#axes[1, 2].set_title(r'$(b)$[Set-1]: $t=30$', fontsize=A, pad=15)
axes[1, 1].set_title('(d): $t=1000$', fontsize=A, pad=5)

axes[1, 1].set_ylim(0, 2*np.pi)
axes[1, 1].set_xlim(0, 2*np.pi)
axes[1, 1].set_aspect(1)

axes[1, 1].set_yticks([0, 2*np.pi])
axes[1, 1].set_xticks([0, 2*np.pi])
axes[1, 1].axes.yaxis.set_visible(False)

axes[1, 1].axes.xaxis.set_visible(True)

label = ["$0$", "$2\pi$"]

axes[1, 1].set_xticklabels(label, fontsize=A)
axes[1, 1].set_yticklabels(label, fontsize=A)


#### Case 3

####

V1_k = hdf5_reader_2d("mode2/Soln_0.000000.h5", "Vkx")
V3_k = hdf5_reader_2d("mode2/Soln_0.000000.h5", "Vkz")

Nx = 512
Nz = 512


kx = np.linspace(0,Nx-1, Nx)
kx[Nx//2+1:Nx] = kx[Nx//2+1:Nx]-Nx

kz = np.linspace(0, Nz//2, Nz//2+1)


kx_mesh, kz_mesh = np.meshgrid(kx, kz,indexing = 'ij')

omega_z = 1j*(kx_mesh*V3_k- kz_mesh*V1_k)

V1 = np.fft.irfft2(V1_k)*(Nx*Nz)
V2 = np.fft.irfft2(V3_k)*(Nx*Nz)

omega_z_r = np.fft.irfft2(omega_z)*(Nx*Nz)

#l_lim = np.min(omega_z_r)
#u_lim = np.max(omega_z_r)

l_lim = np.min(V2)
u_lim = np.max(V2)


#print(l_lim, u_lim, np.sum(np.transpose(omega_z_r)))

#print (np.max(omega_z_r), np.min(omega_z_r))

x = np.linspace(0,2*np.pi, Nx)#, endpoint=False)
z = np.linspace(0,2*np.pi, Nz)#, endpoint=False)

#x_mesh, z_mesh = np.meshgrid(x, z,indexing = 'ij')

#fig, axes = plt.subplots()
g=8

#density = axes[0,1].pcolor(x, z, np.transpose(omega_z_r), cmap='jet')#, shading='auto')#,vmin=-50, vmax=50)

density = axes[0,2].pcolor(x, z, np.transpose(omega_z_r)*(5/60), cmap='jet', vmin=-5, vmax=5, shading='auto')#,vmin=-50, vmax=50)
axes[0,2].quiver(x[::g], z[::g], np.transpose(V1[::g, ::g]), np.transpose(V2[::g, ::g]), units='width', width=0.006)

cb1 = fig.colorbar(density,fraction=0.045, ax=axes[0,2])
cb1.ax.tick_params(labelsize=A)

#print(np.max(np.transpose(omega_z_r)*(5/60)), np.min(np.transpose(omega_z_r)*(5/60)))
#cb1.remove()
#cb1.set_ticks([-60, -30, 0, 30])
#cb1.set_label(r'$\omega$', rotation=0, labelpad=10)

cb1.set_label('$\omega_z$', rotation=0, labelpad=10)

cb1.set_ticks([-5, 0, 5])
cb1.set_ticklabels(["$-5$", "$0$", "$5$"])
cb1.ax.tick_params(labelsize=A)
#axes[0,1].set_xlabel('$x$', fontsize=A)
#axes[0,1].set_ylabel('$y$', fontsize=A)
#axes[0,1].set_title(r'$(c)$[Set-2]: $t=0$', fontsize=A, pad=15)

axes[0, 2].set_title('   Run C \n (e): $t=0$', fontsize=A, pad=5)

axes[0,2].set_aspect(1)
axes[0,2].set_ylim(0, 2*np.pi)
axes[0,2].set_xlim(0, 2*np.pi)

axes[0,2].set_yticks([0, 2*np.pi])
axes[0,2].set_xticks([0, 2*np.pi])
axes[0,2].axes.xaxis.set_visible(False)


axes[0,2].axes.yaxis.set_visible(False)

label = ["$0$", "$2\pi$"]

axes[0,2].set_xticklabels(label, fontsize=A)
axes[0,2].set_yticklabels(label, fontsize=A)



###

V1_k = hdf5_reader_2d("mode2/Soln_999.990000.h5", "Vkx") #"Soln_45.000000.h5", "Vkx")
V3_k = hdf5_reader_2d("mode2/Soln_999.990000.h5", "Vkz") #"Soln_45.000000.h5", "Vkz")

Nx = 512
Nz = 512


kx = np.linspace(0,Nx-1, Nx)
kx[Nx//2+1:Nx] = kx[Nx//2+1:Nx]-Nx

kz = np.linspace(0, Nz//2, Nz//2+1)


kx_mesh, kz_mesh = np.meshgrid(kx, kz,indexing = 'ij')

omega_z = 1j*(kx_mesh*V3_k- kz_mesh*V1_k)

V1 = np.fft.irfft2(V1_k)*(Nx*Nz)
V2 = np.fft.irfft2(V3_k)*(Nx*Nz)

omega_z_r = np.fft.irfft2(omega_z)*(Nx*Nz)

#l_lim = np.min(omega_z_r)
#u_lim = np.max(omega_z_r)

l_lim = np.min(V2)
u_lim = np.max(V2)


#print(l_lim, u_lim, np.sum(np.transpose(omega_z_r)))

#print (np.max(omega_z_r), np.min(omega_z_r))

x = np.linspace(0,2*np.pi, Nx)#, endpoint=False)
z = np.linspace(0,2*np.pi, Nz)#, endpoint=False)

#x_mesh, z_mesh = np.meshgrid(x, z,indexing = 'ij')

#fig, axes = plt.subplots()
g= 34

#density = axes[1, 1].pcolor(x, z, np.transpose(omega_z_r), cmap='jet',vmin=-100, vmax=67)#, shading='auto',vmin=-100, vmax=67)
density = axes[1, 2].pcolor(x, z, np.transpose(omega_z_r)*(12/75), cmap='jet',vmin=-12, vmax=12, shading='auto')#,vmin=-100, vmax=67)
axes[1, 2].quiver(x[::g], z[::g], np.transpose(V1[::g, ::g]), np.transpose(V2[::g, ::g]), units='width', scale=9, scale_units='xy', angles='xy',  width=0.0065)

cb1 = fig.colorbar(density, fraction=0.045, ax=axes[1, 2])
cb1.ax.tick_params(labelsize=A)

#cb1.set_ticks([-80, -40, 0, 40])

#cb1.remove()

cb1.set_ticks([-12, 0, 12])
cb1.set_ticklabels(["$-12$", "$0$", "$12$"])
cb1.ax.tick_params(labelsize=A)

cb1.set_label('$\omega_z$', rotation=0, labelpad=5.8)


axes[1, 2].set_xlabel('$x$', fontsize=A)
#axes[1, 1].set_ylabel('$y$', fontsize=A)
#axes[1, 1].set_title(r'$(d)$[Set-2]: $t=30$', fontsize=A, pad=15)

axes[1, 2].set_title('(f): $t=1000$', fontsize=A, pad=5)

axes[1, 2].set_ylim(0, 2*np.pi)
axes[1, 2].set_xlim(0, 2*np.pi)
axes[1, 2].set_aspect(1)

axes[1, 2].set_yticks([0, 2*np.pi])
axes[1, 2].set_xticks([0, 2*np.pi])
axes[1, 2].axes.yaxis.set_visible(False)

axes[1, 2].axes.xaxis.set_visible(True)

label = ["$0$", "$2\pi$"]

axes[1, 2].set_xticklabels(label, fontsize=A)
axes[1, 2].set_yticklabels(label, fontsize=A)

### Case 3 done

#fig.tight_layout()
#plt.savefig('entropy_analysis_prl/fields_2d.png', dpi=100, bbox_inches="tight")
#plt.savefig('spectrum_flux_time_series/t_500/fields_2d_5.png', dpi=200, bbox_inches="tight")
plt.savefig('fields_2d.pdf', dpi=100, bbox_inches="tight")

plt.show()



