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
A=2.4*9.3#1.5*9.3
font = {'family' : 'serif', 'weight' : 'normal', 'size' : A}
plt.rc('font', **font)

#fig, axes = plt.subplots(nrows=4, ncols=3,figsize = (13, 14))


fig, axes = plt.subplots(figsize = (8, 6))

##### Case 1 #####
data1 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_pdealias/entropy_modeset1_u_0.txt')
data2 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_pdealias/entropy_modeset1_u_20.txt')
data3 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_pdealias/entropy_modeset1_u_39p99.txt')
data4 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_pdealias/entropy_modeset1_u_59p99.txt')
data5 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_pdealias/entropy_modeset1_u_79p99.txt')
data6 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_pdealias/entropy_modeset1_u_104p63.txt')
data8 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_pdealias/entropy_modeset1_u_140.txt')
data9 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_pdealias/entropy_modeset1_u_500.txt')

data14 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_modeset1_u_709.txt')

data15 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_set_a_1000.txt')

data16 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_modeset2_u_1500.txt')
#data7 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_pdealias/entropy_t_3_75.txt')

data7 = np.loadtxt('2d_random_tgv_k11_pdealias_chaos/entropy_pdealias/entropy_t_12_75.txt')


time = data1[:, 0]
entropy = data1[:, 1]

time1 = data2[:, 0]
entropy1 = data2[:, 1]

time2 = data3[:, 0]
entropy2 = data3[:, 1]

time3 = data4[:, 0]
entropy3 = data4[:, 1]


time4 = data5[:, 0]
entropy4 = data5[:, 1]

time5 = data6[:, 0]
entropy5 = data6[:, 1]

time7 = data8[:, 0]
entropy7 = data8[:, 1]


time6 = data7[:, 0]
entropy6 = data7[:, 1]

time9 = data9[:, 0]
entropy9 = data9[:, 1]


time14 = data14[:, 0]
entropy14 = data14[:, 1]


time15 = data15[:, 0]
entropy15 = data15[:, 1]


time16 = data16[:, 0]
entropy16 = data16[:, 1]


#fit = np.polyfit(time6, np.log(entropy6), 1)

#fit2 = np.polyfit(time6, entropy6, 3)


#fit2 = np.polyfit(time6, entropy6, 2)

#fit2 = np.polyfit(time6, entropy6, 4)

#y1 = np.exp(fit[1])*np.exp(fit[0]*time6)

#y2 = fit2[1]*(time6**2)+fit2[0]*(time6**3)+fit2[2]*(time6)+fit2[3]

#y2 = fit2[1]*(time6**3)+fit2[0]*(time6**4)+fit2[2]*(time6**2)+fit2[3]*(time6)+fit2[4]

#y2 = fit2[1]*(time6)+fit2[0]*(time6**2)+fit2[2]

#y1 = fit[0]*(time6)+fit[1]

#print (fit)

axes.plot( time, entropy, 'r', lw=3)
axes.plot( time1, entropy1, 'r', lw=3)
axes.plot( time2, entropy2, 'r', lw=3)
axes.plot( time3, entropy3, 'r', lw=3)
axes.plot( time4, entropy4, 'r', lw=3)
axes.plot( time5, entropy5, 'r', lw=3)
axes.plot( time7, entropy7, 'r', lw=3)
axes.plot( time9, entropy9, 'r', lw=3)
axes.plot( time14, entropy14, 'r', lw=3)
axes.plot( time15, entropy15, 'r', label="Run A", lw=3)
axes.plot( time16, entropy16, 'r', lw=3)

#axes.text(26.6, 6.3, r"$4.9+3.9 e^{-0.05 t}$", color="k", fontsize=A)

fit2 = np.polyfit(time6, np.log(abs(entropy6-(4.92))), 1)

y2 =  np.exp(fit2[1])*np.exp(fit2[0]*time6)+(4.92)

print ("case-1", np.exp(fit2[1]))


#axes.plot( time6[:-8], y2[:-8], 'k', lw=2, linestyle='--')

#axes[0, 0].axvspan(2, 75, color='grey', alpha=0.2, lw=0)

#axes[0, 0].axvspan(10.54, 75, color='grey', alpha=0.2, lw=0)

#axes.axhline(y=4.9, color='k', linestyle = "--", lw=1.7)

#axes.text(90*(30/np.max((time5))), 3.8, r'$S=4.92$', color="red")

axes.text(822, 4.6, r'$S=4.0$', color="k", fontsize=A)

#axes.set_xlim(0, 140)
#axes.set_xticks([0, 40, 80, 120])

axes.set_ylim(0, 11)
#axes.set_yticks([0, 2, 4, 6, 8, 10])

#axes.set_title(' Case-1 \n(a):', fontsize=A, pad=5)


#axes.set_xlabel('$t$')
axes.set_ylabel(r'$S$', fontsize=1.1*A)


### for max calculation

entropy_full_r = np.concatenate((entropy, entropy1, entropy2, entropy3, entropy4, entropy5, entropy7))
time_full_r = np.concatenate((time, time1, time2, time3, time4, time5, time7))
i = np.unravel_index(entropy_full_r.argmax(), entropy_full_r.shape)

print("Run A max entropy", entropy_full_r[i], "index", i, "time", time_full_r[i])

#axes.axhline(y=entropy_full_r[i], xmin = 0, xmax= 10, color='r', linestyle = "--", lw=1.7)

##### Case-2 #####

data1 = np.loadtxt('mode0/entropy_modeset1_u.txt')
data2 = np.loadtxt('mode0/entropy_modeset1_u_1.txt')
data3 = np.loadtxt('mode0/entropy_modeset1_u_2.txt')
data4 = np.loadtxt('mode0/entropy_modeset1_u_3.txt')

data13 = np.loadtxt('mode0/entropy_modeset1_603.txt')
data12 = np.loadtxt('mode0/entropy_modeset1_1000.txt')

data20 = np.loadtxt('mode0/entropy_modeset2_u_1500.txt')

data8 = np.loadtxt('mode0/entropy_t_6_p45_10_p_2.txt')

time8 = data8[:, 0]
entropy8 = data8[:, 1]


time = data1[:, 0]
entropy = data1[:, 1]

time1 = data2[:, 0]
entropy1 = data2[:, 1]

time2 = data3[:, 0]
entropy2 = data3[:, 1]

time3 = data4[:, 0]
entropy3 = data4[:, 1]

time13 = data13[:, 0]
entropy13 = data13[:, 1]

time12 = data12[:, 0]
entropy12 = data12[:, 1]

time20 = data20[:, 0]
entropy20 = data20[:, 1]


#fit3 = np.polyfit(time8, entropy8, 3)

#print ("case-3", fit3)

fit3 = np.polyfit(time8, np.log(abs(entropy8-(1.2))), 1)

y3 =  np.exp(fit3[1])*np.exp(fit3[0]*time8)+(1.2)

print ("case-2", fit3)

entropy_full_r = np.concatenate((entropy, entropy1, entropy2, entropy3, entropy13, entropy12, entropy20))
time_full_r = np.concatenate((time, time1, time2, time3, time13, time12, time20))

#y3 = fit3[1]*(time8**2)+fit3[0]*(time8**3)+fit3[2]*(time8)+fit3[3]

#axes[0, 2].axvspan(6.459999999999906706e+00, 1.100999999999980972e+01, color='grey', alpha=0.2, lw=0)

#axes.plot( time, entropy, 'b', lw = 2, alpha=0.5)
#axes.plot( time1, entropy1, 'b', lw = 2, label="Run C", alpha=0.5)

axes.plot( time_full_r, entropy_full_r, 'b', lw = 2, label="Run B", alpha=0.6)

#axes.plot( time8[:-10], y3[:-10], 'k', lw=2, linestyle='--')
#axes.text(8.0, 1.76, r"$1.2+765e^{-1.03 t}$", color="k", fontsize=A)

#axes.axhline(y=1.18, color='k', linestyle = "--", lw=1.7)

#axes[0, 2].set_xlabel('$t$')
#axes[0, 2].set_ylabel('$S$')

#axes.text(20, 0.88, r'$S=1.2$', color="red")
axes.text(820, 1.3, r'$S=1.05$', color="k", fontsize=A)

#axes[0, 2].set_title(r'Case-2 [Flux [$t=30$]]', fontsize=A, pad=15)

#axes.set_title(' Case-3 \n(i):', fontsize=A, pad=5)

#axes.set_xlim(0, 30)


#axes.set_ylim(0, 3.0)
axes.set_yticks([0, 2, 4, 6, 8, 10])
#yticklabel = ["$0$", "$2$", "$4$", "$6$",  "$8$", "$10$"]

#axes.set_yticklabels(yticklabel, fontsize=A)

#axes.set_xticks([0, 10, 20, 30], fontsize=A)
#xticklabel = ["$0$", "$10$",  "$20$", "$30$"]

#axes.set_xticklabels(xticklabel, fontsize=A)


# for max entropy calculations

i = np.unravel_index(entropy_full_r.argmax(), entropy_full_r.shape)

sec_max_index = np.where(time > 1)
i = np.unravel_index(entropy_full_r[sec_max_index].argmax(), entropy_full_r.shape)

print("Run B max entropy", entropy_full_r[i], "index", i, "time", time_full_r[i])


#axes.axhline(y=entropy_full_r[i], xmin = 0, xmax= 10, color='b', linestyle = "--", lw=1.7)

##### Case 3 #####

data3 = np.loadtxt('mode2/entropy_modeset3_u.txt')
data4 = np.loadtxt('mode2/entropy_modeset3_u_1.txt')
data5 = np.loadtxt('mode2/entropy_modeset3_u_2.txt')
data6 = np.loadtxt('mode2/entropy_modeset3_u_3.txt')
data7 = np.loadtxt('mode2/entropy_modeset3_u_4.txt')
data10 = np.loadtxt('mode2/entropy_modeset3_u_5.txt')
data11 = np.loadtxt('mode2/entropy_modeset3_1000.txt')

time3 = data3[:, 0]
entropy3 = data3[:, 1]

time4 = data4[:, 0]
entropy4 = data4[:, 1]

time5 = data5[:, 0]
entropy5 = data5[:, 1]

time6 = data6[:, 0]
entropy6 = data6[:, 1]

time7 = data7[:, 0]
entropy7 = data7[:, 1]


time10 = data10[:, 0]
entropy10 = data10[:, 1]


time11 = data11[:, 0]
entropy11 = data11[:, 1]


data9 = np.loadtxt('mode2/entropy_t_8p53_15.txt')

time9 = data9[:, 0]
entropy9 = data9[:, 1]

#fit4 = np.polyfit(time9, entropy9, 3)

fit4 = np.polyfit(time9[10:], np.log(abs(entropy9[10:]-(3.1))), 1)

y4 =  np.exp(fit4[1])*np.exp(fit4[0]*time9)+(3.1)

print ("case-3", fit4)

#y4 = fit4[1]*(time9**2)+fit4[0]*(time9**3)+fit4[2]*(time9)+fit4[3]


#axes[0, 1].axvspan(7.899999999999876010e+00, 1.500999999999972445e+01, color='grey', alpha=0.2, lw=0)

#axes[0, 1].loglog(time3[1:], entropy3[1:], 'r', lw = 1)
#axes[0, 1].loglog(time4, entropy4, 'r', lw = 1)

entropy_full_r_2 = np.concatenate((entropy3, entropy4, entropy5, entropy6, entropy7, entropy10,entropy11))
time_full_r_2 = np.concatenate((time3, time4, time5, time6, time7, time10, time11))

#axes.plot(time3, entropy3, 'g', lw = 2)
#axes.plot(time4, entropy4, 'g', lw = 2, label="Run B")
axes.plot(time_full_r_2, entropy_full_r_2, 'g', lw = 1, label="Run C", alpha=0.8)
#axes.plot( time9[10:], y4[10:], 'k', lw=2, linestyle='--')

#axes.axhline(y=3.07, color='k', linestyle = "--", lw=1.7)

#axes.text(10.8, 3.8, r"$3.1+365e^{-0.63 t}$", color="k", fontsize=A)
axes.set_xlabel(r"$t$", fontsize=1.1*A)
#axes.set_ylabel('$S$', labelpad=35)

axes.set_xlim(0, 1000)


#axes.text(22, 2.5, r'$S=3.1$', color="r")

axes.text(820, 3.0, r'$S=2.6$', color="k", fontsize=A)


# for max calculation

i = np.unravel_index(entropy_full_r_2.argmax(), entropy_full_r_2.shape)

print("Run C max entropy", entropy_full_r_2[i], "index", i, "time", time_full_r_2[i])

#axes.axhline(y=entropy_full_r_2[i], xmin = 0, xmax= 10, color='g', linestyle = "--", lw=1.7)
#####
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

#axes.set_title(' Case-2 \n(e):', fontsize=A, pad=5)

#axes.set_ylim(0, 6.0)
#axes[0, 1].set_yticks([0, 2, 4, 6])
#axes[0, 1].set_xticks([0, 10, 20, 30])


plt.legend(loc='best', fontsize=A, frameon=False)

fig.tight_layout()
plt.savefig('spectrum_flux_time_series/t_500/entropy.pdf', dpi=600)#, bbox_inches="tight")

plt.show()
