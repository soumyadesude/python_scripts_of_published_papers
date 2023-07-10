import numpy as np
import scipy as sp
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
plt.rcParams['xtick.major.size'] = 1.5*4.0
plt.rcParams['xtick.major.width'] = 2*0.5
plt.rcParams['xtick.minor.size'] = 1.5*2.5
plt.rcParams['xtick.minor.width'] = 2*0.5
plt.rcParams['ytick.major.size'] = 1.5*4.0
plt.rcParams['ytick.major.width'] = 2*0.5
plt.rcParams['ytick.minor.size'] = 1.5*2.5
plt.rcParams['ytick.minor.width'] = 2*0.5
A=2.3*9.3#1.5*9.3
font = {'family' : 'serif', 'weight' : 'normal', 'size' : A}
plt.rc('font', **font)

#fig, axes = plt.subplots(nrows=4, ncols=3,figsize = (13, 14))


#fig, axes = plt.subplots(1, 2, figsize = (18, 6),  gridspec_kw={'wspace':0.28, 'hspace':0})

fig, axes = plt.subplots(1, 2, figsize = (13, 5))

## Run A
data1 = np.loadtxt('RUN_A/enstrophy/nohup_1u.out')
data2 = np.loadtxt('RUN_A/enstrophy/nohup_2u.out')
data3 = np.loadtxt('RUN_A/enstrophy/nohup_3u.out')
data4 = np.loadtxt('RUN_A/enstrophy/nohup_4u.out')
data5 = np.loadtxt('RUN_A/enstrophy/nohup_5u.out')
data6 = np.loadtxt('RUN_A/enstrophy/nohup_6u.out')
data7 = np.loadtxt('RUN_A/enstrophy/nohup_7u.out')


time = data1[:, 0]
energy = data1[:, 1]

time1 = data2[:, 0]
energy1 = data2[:, 1]

time2 = data3[:, 0]
energy2 = data3[:, 1]

time3 = data4[:, 0]
energy3 = data4[:, 1]


time4 = data5[:, 0]
energy4 = data5[:, 1]

time5 = data6[:, 0]
energy5 = data6[:, 1]

time6 = data7[:, 0]
energy6 = data7[:, 1]

time = np.concatenate((time, time1, time2, time3, time4, time5, time6))
energy = np.concatenate((energy, energy1, energy2, energy3, energy4, energy5, energy6))



#axes[0].plot( time*(30/np.max((time5))), energy, 'r', lw=3)
#axes[0].plot( time1*(30/np.max((time5))), energy1, 'r', lw=3)
#axes[0].plot( time2*(30/np.max((time5))), energy2, 'r', lw=3)
#axes[0].plot( time3*(30/np.max((time5))), energy3, 'r', lw=3)
#axes[0].plot( time4*(30/np.max((time5))), energy4, 'r', lw=3)
#axes[0].plot( time5*(30/np.max((time5))), energy5, 'r', label="Run A", lw=3)

axes[0].plot( time*(30/np.max((time))), energy, 'r', label="Run A", lw=3)


# error

#axes[1].plot( time*(30/np.max((time5))), abs((energy-energy[0])/energy[0]), 'r', lw=3)
#axes[1].plot( time1*(30/np.max((time5))), abs((energy1-energy[0])/energy[0]), 'r', lw=3)
#axes[1].plot( time2*(30/np.max((time5))), abs((energy2-energy[0])/energy[0]), 'r', lw=3)
##axes[1].plot( time3*(30/np.max((time5))), abs((energy3-energy[0])/energy[0]), 'r', lw=3)
#axes[1].plot( time4*(30/np.max((time5))), abs((energy4-energy[0])/energy[0]), 'r', lw=3)
#axes[1].plot( time5*(30/np.max((time5))), abs((energy5-energy[0])/energy[0]), 'r', label="Run A", lw=3)


axes[1].plot( time*(30/np.max((time))), abs((energy-energy[0])/energy[0]), 'r', label="Run A", lw=3)


print("Run A", abs((energy[-1]-energy[0])/energy[0]))
## Run B
data1_1 = np.loadtxt('RUN_B/enstrophy/nohup_1u.out')
data2_1 = np.loadtxt('RUN_B/enstrophy/nohup_2u.out')

time_1 = data1_1[:, 0]
energy_1 = data1_1[:, 1]

time1_1 = data2_1[:, 0]
energy1_1 = data2_1[:, 1]

time = np.concatenate((time_1, time1_1))
energy = np.concatenate((energy_1, energy1_1))


#axes[0].plot( time_1, energy_1, 'b', lw=2)
#axes[0].plot( time1_1, energy1_1, 'b', label="Run B", lw=2)

axes[0].plot( time, energy, 'b--', label="Run B", lw=2, alpha=0.6)

# error
#axes[1].plot( time_1, abs((energy_1-energy_1[0])/energy_1[0]), 'b', lw=2)
#axes[1].plot( time1_1, abs((energy1_1-energy_1[0])/energy_1[0]), 'b', label="Run B", lw=2)

axes[1].plot( time, abs((energy-energy[0])/energy[0]), 'b--', label="Run B", lw=2, alpha=0.6)

print("Run B", abs((energy[-1]-energy[0])/energy[0]))

## Run C
data1_2 = np.loadtxt('RUN_C/enstrophy/nohup_1u.out')
data2_2 = np.loadtxt('RUN_C/enstrophy/nohup_2u.out')
data3_2 = np.loadtxt('RUN_C/enstrophy/nohup_3u.out')
data4_2 = np.loadtxt('RUN_C/enstrophy/nohup_4u.out')
data5_2 = np.loadtxt('RUN_C/enstrophy/nohup_5u.out')


time_2 = data1_2[:, 0]
energy_2 = data1_2[:, 1]

time1_2 = data2_2[:, 0]
energy1_2 = data2_2[:, 1]

time2_2 = data3_2[:, 0]
energy2_2 = data3_2[:, 1]

time3_2 = data4_2[:, 0]
energy3_2 = data4_2[:, 1]


time4_2 = data5_2[:, 0]
energy4_2 = data5_2[:, 1]

time = np.concatenate((time_2, time1_2, time2_2, time3_2, time4_2))
energy = np.concatenate((energy_2, energy1_2, energy2_2, energy3_2, energy4_2))

#axes[0].plot( time_2*(30/np.max((time4_2))), 0.1*energy_2, 'g', lw=1)
#axes[0].plot( time1_2*(30/np.max((time4_2))), 0.1*energy1_2, 'g', lw=1)
#axes[0].plot( time2_2*(30/np.max((time4_2))), 0.1*energy2_2, 'g', lw=1)
#axes[0].plot( time3_2*(30/np.max((time4_2))), 0.1*energy3_2, 'g', lw=1)
#axes[0].plot( time4_2*(30/np.max((time4_2))), 0.1*energy4_2, 'g',  label="Run C", lw=1)
axes[0].plot( time*(30/np.max((time))), energy, 'g',  label="Run C", lw=1, alpha=0.8)


#error
##axes[1].plot( time_2*(30/np.max((time4_2))), abs((energy_2-energy_2[0])/energy_2[0]), 'g', lw=1)
#axes[1].plot( time1_2*(30/np.max((time4_2))),abs((energy1_2-energy_2[0])/energy_2[0]), 'g', lw=1)
#axes[1].plot( time2_2*(30/np.max((time4_2))), abs((energy2_2-energy_2[0])/energy_2[0]), 'g', lw=1)
##axes[1].plot( time3_2*(30/np.max((time4_2))), abs((energy3_2-energy_2[0])/energy_2[0]), 'g', lw=1)
#axes[1].plot( time4_2*(30/np.max((time4_2))), abs((energy4_2-energy_2[0])/energy_2[0]), 'g',  label="Run C", lw=1)
axes[1].plot( time*(30/np.max((time))), abs((energy-energy[0])/energy[0]), 'g',  label="Run C", lw=1, alpha=0.8)

print("Run C", abs((energy[-1]-energy[0])/energy[0]))

axes[1].set_yscale("log")


axes[0].set_xlim(0, 30)
axes[0].set_xticks([0, 10, 20, 30], fontsize=A)

axes[1].set_xlim(0, 30)
axes[1].set_xticks([0, 10, 20, 30], fontsize=A)


#axes[0].set_ylim(0, 80)
#axes[0].set_yticks([0, 20, 40, 60, 80], fontsize=A)

axes[1].set_ylim(1e-15, 1e-5)
axes[1].set_yticks([1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5], fontsize=A)


axes[0].set_ylabel("$\Omega}(t)$", fontsize=A)
axes[0].set_xlabel("$t'$", fontsize=A)

axes[0].set_yscale("log")
axes[1].set_ylabel(r"$\epsilon_{\Omega}$", fontsize=1.1*A)
axes[1].set_xlabel("$t'$", fontsize=A)

axes[1].text(2, 2e-15,  '(b)')
axes[0].text(2, 1.3e0,  '(a)')


#plt.legend(bbox_to_anchor =(0.45, 1.25), fontsize=A, frameon=False, ncol=3)
axes[1].legend(fontsize=A/1.12, ncol=1, scatterpoints=1, loc="lower right", frameon=False)

plt.tight_layout()
plt.savefig('enstrophy_time.pdf', dpi=600, bbox_inches="tight")
plt.show()



