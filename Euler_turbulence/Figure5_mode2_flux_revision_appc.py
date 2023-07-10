import numpy as np
import scipy as sp

from scipy import signal
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
mpl.style.use('classic')
plt.rcParams['xtick.major.size'] = 1.5*4.0
plt.rcParams['xtick.major.width'] = 2*0.5
plt.rcParams['xtick.minor.size'] = 1.5*2.5
plt.rcParams['xtick.minor.width'] = 2*0.5
plt.rcParams['ytick.major.size'] = 1.5*4.0
plt.rcParams['ytick.major.width'] = 2*0.5
plt.rcParams['ytick.minor.size'] = 1.5*2.5
plt.rcParams['ytick.minor.width'] = 2*0.5
#A=1.05*((2.0*9.3)/1.22)#1.5*9.3
A=2.3*9.3
font = {'family' : 'serif', 'weight' : 'normal', 'size' : A}
plt.rc('font', **font)

######

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#######
#fig, axes = plt.subplots(nrows=2, ncols=1,figsize = (7, 9), sharex=True, gridspec_kw={'wspace':0, 'hspace':0})
#fig, axes = plt.subplots(nrows=2, ncols=1,figsize = (7, 9))#, gridspec_kw={'wspace':0, 'hspace':0})
fig, axes = plt.subplots(2, 1, figsize = (7,9))#,  gridspec_kw={'wspace':0.28, 'hspace':0})

#fig, axes = plt.subplots(nrows=2, ncols=1,figsize = (7, 10), gridspec_kw={'wspace':0, 'hspace':0.2})


#### Case 1 ####
input0 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/time_averaged_spectrum_correct.txt')
#input0_0 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/time_averaged_spectrum1.txt')

k = np.arange(0,len(input0[:,0]),1)

E = input0[:, 1]
#E1 = input0_0[:, 1]

E0 = E #(E+E1)/2

x1 = np.linspace(18,171,3000)
x2 = np.linspace(2,12,3000)

x = k[1:171]
y = E0[1:171]

def fit_spec(x, a, b):
	return (a+(b*x**2))

param, param_cov = curve_fit(fit_spec, x, x/y)

print("RUNA", param)

fit = x/(param[0]+(param[1]*x**2))
#fit = (param[0]+(param[1]*x**2))
#axes[0].loglog(k,(E0/0.25), 'r', lw=2, label='Run A')
axes[0].semilogy(k, E0, 'r', lw=3, label='Run A')
axes[0].semilogy(x[10:], fit[10:], 'k--', lw=2)
#axes[0].loglog(x1,(2.4e-3)*(x1**(-1)),'k-',linewidth=0.75)
#axes[0,3].loglog(x2,(5.0e-1)*(x2**(-3)),'k-',linewidth=0.75)
#axes[0].loglog(x2,(3e-1)*(x2**(-3)),'k-',linewidth=0.75)

#axes[0].text(50, 1e-5, r'$\sim k^{-1}$')
#axes[0].text(2, 1e-3, r'$\sim k^{-3}$')

#axes[0,3].axvline(x=np.ceil(11*np.sqrt(2)), color='k', linestyle = "--")
#axes[0].vlines(x=15, ymin = 5e-7, ymax = 3e-4, color='k', linestyle = "--")

#axes[0].set_ylabel('$E(k)/E$' , fontsize=A)
#axes[0].set_xlabel('$k$' , fontsize=A)



#left, bottom, width, height = [0.40, 0.61, 0.38, 0.34]

#ax4 = axes[0].inset_axes([left, bottom, width, height])

#ax4.loglog(k, E0, 'r', lw=3, label='Run A')
#ax4.loglog(k, 1e-1*k**(-5/3), 'k--', lw=2, label='Run A')


#ax4.set_ylabel('$E(k)$' , fontsize=A/1.4)
#ax4.set_xlabel('$k$' , fontsize=A/1.4, labelpad=0.5)


#axes[0,3].set_title('$(a):$ $t=120$', fontsize=A, pad=15)

#axes[0].set_title('(a):', fontsize=A, pad=5)

#axes[0].set_yticks([1e-6, 1e-4,  1e-2, 1e0])

axes[0].set_xlim(1,171)



#axes[0].axes.xaxis.set_visible(False)

flux = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/time_averaged_flux_correct.txt')
#flux_1 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/time_averaged_flux1.txt')

flux1 = flux #(flux+flux_1)/2

k = np.arange(0,len(flux1),1)

#axes[1].plot(k,(flux1[::1]/(0.25)**(3/2)), 'r', lw=2, label='$t=120$', marker='s', markersize=5, markerfacecolor='r', markeredgecolor='r', markeredgewidth=0.8)
axes[1].plot(k,(flux1[::1]), 'r', lw=1, label='Run A', marker='s', markersize=5, markerfacecolor='r', markeredgecolor='r', markeredgewidth=0.8)
#axes[1].plot(k,(smooth(flux1[::1],2)/(0.25)**(3/2)), 'r', lw=2, label='$t=120$', marker='s', markersize=5, markerfacecolor='r', markeredgecolor='r', markeredgewidth=0.8)

axes[1].set_xlim(1, 20)

#axes[1].set_xscale('log')
#axes[1].set_yscale('symlog', linthresh=1e-3)


#axes[1].set_ylim(-5e-3, 1e-2)
#axes[1].set_yticks([-1e-3, 0,  1e-3])

##ax2.legend(loc='upper right', fontsize=A/1.37, frameon=True)

##ax2.axvline(x=np.ceil(11*np.sqrt(2)), color='k', linestyle = "--")
##ax2.axvline(x=15, color='k', linestyle = "--")


axes[1].set_ylabel('$\Pi(k)$' , fontsize=A/1.1)
axes[1].set_xlabel('$k$' , fontsize=A)
#axes[0].set_xlabel('$k$' , fontsize=A)

#axes[1].axhline(y=0, color='k')

#plt.ticklabel_format( axis='y',  style='sci')
#axes[1].set_title('$(b):$ $t=120$', fontsize=A, pad=15)

left, bottom, width, height = [0.61, 0.2, 0.36, 0.37]

ax2 = axes[1].inset_axes([left, bottom, width, height])

ax2.plot(k,(flux1[::1]), 'r', lw=3, label='Run A', marker='s', markersize=0, markerfacecolor='r', markeredgecolor='r', markeredgewidth=0.8)

##ax2.set_xlim(1, 10)

print("min upto 10: RUN A", np.min(flux1[1:10:1]))

##ax2.set_yscale('symlog', linthresh=1e-3)

#ax2.set_ylabel('$\Pi(k)$' , fontsize=A/1.4)
#ax2.set_xlabel('$k$' , fontsize=A/1.4)




### Case 1 done



## Case 2

input0 = np.loadtxt('mode0/time_averaged_spectrum.txt')


k = np.arange(0,len(input0[:,0]),1)

E0 = input0[:, 1]


x1 = np.linspace(40,171,3000)
x2 = np.linspace(5,22,3000)

#axes[0].loglog(k,(E0/4), 'b', lw=2, label='Run C ')
axes[0].semilogy(k,(E0), 'b', lw=2, label='Run B', alpha=0.6)


#ax4.loglog(k,(E0), 'b', lw=2, label='Run B', alpha=0.6)

x = k[1:171]
y = E0[1:171]

param, param_cov = curve_fit(fit_spec, x, x/y)

print("RUNB", param)

fit = x/(param[0]+(param[1]*x**2))
#fit = (param[0]+(param[1]*x**2))
axes[0].semilogy(x[40:], fit[40:], 'k', lw=2, linestyle='--')




x1 = np.linspace(50,171,3000)

#axes[0].semilogy(x1,(9.1e-2)*(x1**(-1)),'k-',linewidth=1)
axes[0].semilogy(x1,(3.1e-4)*(x1**(-1)),'k-',linewidth=1)

#axes[0].text(100, 2e-3, r'$\sim k^{-1}$', fontsize=A)

axes[0].text(100, 4.6e-6, r'$ k^{-1}$', fontsize=A)

#axes[1, 0].loglog(x1,(6.1e-3)*(x1**(-1)),'k-',linewidth=0.75)

#axes[0].loglog(x1,(0.34e-3)*x1**(-1.1),'k-',linewidth=0.75)

#axes[0,3].loglog(x2,(5.0e-1)*(x2**(-3)),'k-',linewidth=0.75)
#axes[1, 0].loglog(x2,(1.2e0)*(x2**(-3)),'k-',linewidth=0.75)


#axes[0].loglog(x2,(0.52e-1)*x2**(-2.8),'k-',linewidth=0.75)
#axes[0].loglog(x2,(0.2e-1)*x2**(-2.8),'k-',linewidth=0.75)

#axes[0].text(60, 8e-6, r'$\sim k^{-1.1}$')
#axes[0].text(4, 5e-6, r'$\sim k^{-2.8}$')

#axes[0,3].axvline(x=np.ceil(11*np.sqrt(2)), color='k', linestyle = "--")
#axes[1, 0].axvline(x=15, color='k', linestyle = "--")

#axes[0].set_ylabel('$E(k)$' , fontsize=A)
#axes[0].set_xlabel('$k$' , fontsize=A)

#axes[0].legend(loc='lower left', fontsize=A, frameon=True)

#axes[0,3].set_title('$(a):$ $t=120$', fontsize=A, pad=15)

#axes[0].set_title('(g): $t=30$', fontsize=A, pad=5)

#axes[0].set_yticks([1e-6, 1e-4,  1e-2, 1e0])

axes[0].set_xlim(1,171)
#axes[1, 0].set_ylim(4e-7, 8e-1)

#axes[0].set_yticks([1e-6, 1e-4,  1e-2, 1e0])
#axes[0].set_yticklabels(["$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$10^{0}$" ], fontsize=A)

#axes[0].set_xticks([1, 1e1,  1e2])
#axes[0].set_xticklabels(["$10^{0}$", "$10^{1}$", "$10^{2}$"], fontsize=A)

axes[0].set_xlim(1,171)
#axes[0].set_ylim(-1e6, 8e6)
#axes[0].ticklabel_format(axis='y',  scilimits=(0,0), useOffset=None, useLocale=None, useMathText=True)


## flux 

flux1 = np.loadtxt('mode0/time_averaged_flux.txt')

k = np.arange(0,len(flux1),1)

#axes[1].plot(k,(flux1[::1]/(4)**(3/2)), 'b', lw=2, label='$t=30$', marker='p', markersize=5, markerfacecolor='b', markeredgecolor='b', markeredgewidth=0.8)
axes[1].plot(k,(flux1[::1]), 'b', lw=1, label='Run B', marker='p', markersize=5, markerfacecolor='b', markeredgecolor='b', markeredgewidth=0.8, alpha=0.6)

ax2.plot(k,(flux1[::1]), 'b', lw=2, label='Run B', marker='p', markersize=0, markerfacecolor='b', markeredgecolor='b', markeredgewidth=0.8, alpha=0.6)

##ax2.set_yscale('symlog', linthresh=1e-4)


print("min upto 10: RUN B", np.min(flux1[1:10:1]))

#axes[1].set_xticklabels(["$0$", "$40$", "$80$", "$120$", "$160$" ], fontsize=A)

#axes[1].set_xscale('log')
#axes[1].set_yscale('symlog', linthresh=1e-3) #linthresh=0.015, linscaley=0.7)


#axes[1].set_ylim(-9e-4, 9e-4)
#axes[1].set_yticks([-1e-4, 0,  1e-4])


#axes[1].set_yticklabels(["$-10^{-3}$", "$-10^{-4}$", "0", "$10^{-4}$", "$10^{-3}$" ], fontsize=A)

#axes[1].axvline(x=np.ceil(11*np.sqrt(2)), color='k', linestyle = "--")
#axes[1].axvline(x=15, color='k', linestyle = "--")


#axes[1].set_ylabel('$\Pi(k)/E^{3/2}$' , fontsize=A/1.1, labelpad=1)
axes[1].set_ylabel('$\Pi(k)$' , fontsize=A/1.1)
#axes[1].set_xlabel('$k$' , fontsize=A)
#axes[1].legend(loc='best', fontsize=A, frameon=False)

#axes[1].legend(loc='lower right', fontsize=A/1.37, frameon=True)

#axes[1].axhline(y=0, color='k')

#axes[1].text(14, 5e-4,  '(b)')
#axes[0].text(1.5, 1e1,  '(a)')

### Case 2 done

### Case 3 

input0 = np.loadtxt('mode2/time_averaged_spectrum_correct.txt')
#input1 = np.loadtxt('mode2/time_averaged_spectrum2.txt')
#input2 = np.loadtxt('mode2/time_averaged_spectrum3.txt')
#input3 = np.loadtxt('mode2/time_averaged_spectrum4.txt')

k = np.arange(0,len(input0[:,0]),1)

E0 = input0[:, 1]
#E1 = input1[:, 1]
#E2 = input2[:, 1]
#E3 = input3[:, 1]

E_avg = E0 #(E0+E1+E2+E3)/4

x1 = np.linspace(15,171,3000)
x2 = np.linspace(4,13,3000)

#axes[0].loglog(k[2:],(E0[2:]/4), 'g', lw=2, label='Run B')
#axes[0].semilogy(k[2:],E0[2:], 'g', lw=1, label='Run C', alpha=0.8)
#axes[0].semilogy(k[2:],E_avg[2:], 'g', lw=1, label='Run C', alpha=0.8)


#ax4.loglog(k[2:],E_avg[2:], 'g', lw=1, label='Run C', alpha=0.8)

#ax4.set_xlim(1, 10)

#ax4.set_ylim(1e-5, 1e1)


#ax4.set_xticks([1, 5, 10], fontsize=A/1.4)


#ax4.set_yticks([1e-5, 1e-3,  1e-1, 1e1], fontsize=A/1.4)

#ax4.tick_params(axis='x', labelsize=A/1.6)
#ax4.tick_params(axis='y', labelsize=A/1.6)


x = k[2:171]
y = E_avg[2:171]

param, param_cov = curve_fit(fit_spec, x, x/y)

print("RUNC", param)

fit = x/(param[0]+(param[1]*x**2))
#fit = (param[0]+(param[1]*x**2))
#axes[0].semilogy(x[10:], fit[10:], 'k', lw=2, linestyle='--')
#axes[0].loglog(x2,(6.1e0)*(x2**(-3)),'k-',linewidth=0.75)
#axes[0].loglog(x2,(1.8e0)*(x2**(-3)),'k-',linewidth=0.75)

#axes[0].loglog(x1,(4e-2)*x1**(-1),'k-',linewidth=0.75)i
#axes[0].loglog(x1,(1.8e-2)*x1**(-1),'k-',linewidth=0.75)

#axes[0,3].loglog(x2,(5.0e-1)*(x2**(-3)),'k-',linewidth=0.75)
#axes[1, 0].loglog(x2,(1.2e0)*(x2**(-3)),'k-',linewidth=0.75)


#axes[1, 1].loglog(x2,(0.86e-1)*x2**(-2.8),'k-',linewidth=0.75)
#axes[0].loglog(x2,(0.83e-2)*x2**(-2.8),'k-',linewidth=0.75)

#axes[0].text(16, 2e-4, r'$\sim k^{-1}$')
#axes[0].text(4, 1e-3, r'$\sim k^{-3}$')

axes[0].axvline(x=10, ymin=0, ymax=0.52, color='k', linestyle = "--", lw=1)
axes[0].axvline(x=40,ymin=0, ymax=0.24, color='k', linestyle = "--", lw=1)#, dashes=(3,1, 2, 3))

#axes[1].axvline(x=10, color='k', linestyle = "--", lw=1)
#axes[1].axvline(x=40, color='k', linestyle = "--", lw=1)#, dashes=(3,1, 2, 3))


#axes[1, 0].axvline(x=15, color='k', linestyle = "--")

#axes[0].set_ylabel('$E(k)$' , fontsize=A)
#axes[0].set_xlabel('$k$' , fontsize=A)

#axes[0].legend(loc='lower right', fontsize=A, frameon=True)

#axes[0,3].set_title('$(a):$ $t=120$', fontsize=A, pad=15)

#axes[0].set_title('(k): $t=30$', fontsize=A, pad=5)

#axes[0].set_yticks([1e-4,  1e-2, 1e0])

#axes[0].set_ylim([1e-5,  1.1e1])

axes[0].set_xlim(1,170)
axes[0].set_xticks([1, 40, 80, 120, 160], fontsize=A)
#axes[0].set_xticklabels(["", "", "", "", ""], fontsize=A)

axes[0].set_ylabel('$E(k)$' , fontsize=A, labelpad=26)

axes[0].set_ylim(1e-7, 1e0)
axes[0].set_yticks([1e-6, 1e-4,  1e-2, 1e0], fontsize=A)


# flux

flux2 = np.loadtxt('mode2/time_averaged_flux_correct.txt')
#flux3 = np.loadtxt('mode2/time_averaged_flux2.txt')
#flux4 = np.loadtxt('mode2/time_averaged_flux3.txt')
#flux5 = np.loadtxt('mode2/time_averaged_flux4.txt')

flux_avg = flux2 #(flux2+flux3+flux4+flux5)/4

k = np.arange(0,len(flux2),1)

#axes[1].plot(k,(flux2[::1]/(4)**(3/2)), 'green', lw=2, label='$t=30$', marker='o', markersize=5, markerfacecolor='limegreen', markeredgecolor='limegreen', markeredgewidth=0.8, alpha=0.8)
#axes[1].plot(k,(smooth(flux2[::1],2)/(4)**(3/2)), 'green', lw=2, label='$t=30$', marker='o', markersize=5, markerfacecolor='green', markeredgecolor='green', markeredgewidth=0.8, alpha=0.8)
#axes[1].plot(k,(flux2[::1]), 'g', lw=2, label='$t=30$', marker='o', markersize=5, markerfacecolor='limegreen', markeredgecolor='limegreen', markeredgewidth=0.8, alpha=0.8)
#axes[1].plot(k,(flux2[::1]), 'g', lw=1, label='Run C', marker='o', markersize=5, markerfacecolor='limegreen', markeredgecolor='limegreen', markeredgewidth=0.8, alpha=0.8)
#axes[1].plot(k,(flux_avg[::1]), 'g', lw=1, label='Run C', marker='o', markersize=5, markerfacecolor='limegreen', markeredgecolor='limegreen', markeredgewidth=0.8, alpha=0.8)


#ax2.plot(k,(flux_avg[::1]), 'g', lw=1, label='Run C', marker='o', markersize=0, markerfacecolor='limegreen', markeredgecolor='limegreen', markeredgewidth=0.8, alpha=0.8)


#ax2.set_yscale('symlog', linthresh=1.2e-4)

print("min upto 10: RUN C", np.min(flux_avg[1:10:1]))

axes[1].set_xlim(1, 10)
axes[1].set_xticks([1, 3, 5, 7, 9])
#axes[1].set_xscale('log')
#axes[1].set_yscale('symlog', linthresh=1e-3)#linthresh=0.015, linscaley=0.7)


#axes[1].set_ylim(-9e-4, 9e-4)
#axes[1].set_yticks([-1e-4, 0,  1e-4])

#axes[1].set_ylim(-9.5e-1, 2e-1)
#axes[1].set_yticks([-1e-1, 0, 1e-1])


#axes[1].axvline(x=np.ceil(11*np.sqrt(2)), color='k', linestyle = "--")
#axes[1].axvline(x=15, color='k', linestyle = "--")


axes[1].set_ylabel('$\Pi(k)$' , fontsize=A, labelpad=12)

axes[1].set_ylim(-0.0016, 0.0004)
axes[1].set_yticks([-0.0016, -0.0008, 0, 0.0004])

axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
axes[1].yaxis.offsetText.set_fontsize(A)


ax2.set_ylim(-1.1e-3, 4e-4)
ax2.set_yticks([ -8e-4, -4e-4, 0, 4e-4  ], fontsize=A/1.4)
#ax2.set_yticklabels([ "", "\N{MINUS SIGN}10$^{-4}$", "", "", "0", "", "","10$^{-4}$", ""], fontsize=A/1.4)

#axes[1].set_yticklabels(["\N{MINUS SIGN}10$^{-3}$", "", "", "0", "", "","10$^{-3}$"], fontsize=A)


ax2.set_xlim(1, 170)
ax2.set_xticks([1,  80,  160], fontsize=A/1.4)


#ax2.set_ylim(-1e-3, 1e-3)
#ax2.set_yticks([-1e-3, -5e-4, -1e-4, 0, 5e-4, 1e-4, 1e-3])
#ax2.set_yticklabels(["\N{MINUS SIGN}10$^{-3}$", "", "", "0", "", "","10$^{-3}$"], fontsize=A/1.4)
#ax2.set_xlim(1, 171)
##ax2.set_xticks([1, 10], fontsize=A/1.4)

axes[1].axhline(y=0, lw=1, linestyle="--", color="k")

ax2.set_xlabel('$k$' , fontsize=A/1.4)

ax2.set_ylabel('$\Pi(k)$' , fontsize=A/1.4)

ax2.tick_params(axis='both', labelsize=A/1.4)

ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
ax2.yaxis.offsetText.set_fontsize(A/1.4)


#axes[1].legend(loc='best', fontsize=A, frameon=False)

#axes[1].legend(loc='lower right', fontsize=A/1.37, frameon=True)

#axes[1].axhline(y=0, color='k')

### Case 3 done


axes[1].text(9, 1.7e-4,  '(b)')
axes[0].text(150, 0.15,  '(a)')

#axes[0].legend(loc='lower left', fontsize=A, frameon=True)
axes[1].legend(bbox_to_anchor =(-0.03, -0.03), fontsize=A/1.12, ncol=1, scatterpoints=1, loc="lower left", frameon=False)

#pop_a = mpatches.Patch(color='r', label='Run A')
#pop_b = mpatches.Patch(color='b', label='Run B', alpha=0.6)
#pop_c = mpatches.Patch(color='g', label='Run C', alpha= 0.8)
#plt.legend(fontsize=A/1.2, ncol=1, loc="lower left", frameon=False, handles=[pop_a,pop_b, pop_c])

fig.tight_layout()
plt.savefig('appc_2d_figs/spectrum_flux_2d_neq.pdf', dpi=600, bbox_inches="tight")

plt.show()
