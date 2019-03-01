import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Epsilon = np.linspace(0, 0.2, 10)
Epsilon = [0.0, 0.0222222222222, 0.0444444444444, 0.0666666666667, 0.0888888888889, 0.111111111111, 0.133333333333, 0.155555555556, 0.177777777778, 0.2]

# Loading all the attack values
PGD_FGSM = [];
PGD_GD = [];
PGD_CW = [];
SDP_FGSM = [];
SDP_GD = [];
SDP_CW = [];

for e in Epsilon:
    pgd = np.load('attacks-pgd/pgd_results/' + 'FGSM-01-' + str(e) + '.npy')
    PGD_FGSM.append(np.sum(pgd)/np.size(pgd))
    pgd = np.load('attacks-pgd/pgd_results/' + 'GD-01-' + str(e) + '.npy')
    PGD_GD.append(np.sum(pgd)/np.size(pgd))
    pgd = np.load('attacks-pgd/pgd_results/' + 'CW-01-' + str(e) + '.npy')
    PGD_CW.append(pgd)

    sdp= np.load('attacks-sdp/sdp_results/' + 'FGSM-01-' + str(e) + '.npy')
    SDP_FGSM.append(np.sum(sdp)/np.size(sdp))
    sdp = np.load('attacks-sdp/sdp_results/' + 'GD-01-' + str(e) + '.npy')
    SDP_GD.append(np.sum(sdp)/np.size(sdp))
    sdp = np.load('attacks-sdp/sdp_results/' + 'CW-01-' + str(e) + '.npy')
    SDP_CW.append(sdp)


# Loading all the bounds
PGD_SPE = []
PGD_FRO = []
SDP_SPE = []
SDP_FRO = []
SDP_SDP = []

# Changed Epsilon, otherwise file not found
EpsilonStr = ['0.0', '0.022222222222222223', '0.044444444444444446', '0.06666666666666667', '0.08888888888888889', '0.11111111111111112', '0.13333333333333333', '0.15555555555555556', '0.17777777777777778', '0.2']

for e in EpsilonStr:
    #pgd = np.load('bounds-pgd/pgd_results/Fro-01-' + e + '.npy')
    #PGD_FRO.append(np.sum(pgd)/np.size(pgd))
    #pgd = np.load('bounds-pgd/pgd_results/Spectral-01-' + e + '.npy')
    #PGD_SPE.append(np.sum(pgd)/np.size(pgd))

    sdp = np.load('bounds-sdp/sdp_results/Fro-01-' + e + '.npy')
    SDP_FRO.append(np.sum(sdp)/np.size(sdp))
    sdp = np.load('bounds-sdp/sdp_results/Spectral-01-' + e + '.npy')
    SDP_SPE.append(np.sum(sdp)/np.size(sdp))
    sdp = np.load('bounds-sdp/sdp_results/SDP-01-' + e + '.npy')
    SDP_SDP.append(np.sum(sdp)/np.size(sdp))


# Plotting on one axis for PGD network
#plt.plot(Epsilon, PGD_SPE, '#911eb4', label = 'Spectral', linewidth = 3)
#plt.plot(Epsilon, PGD_FRO, 'c', label = 'Frobenius', linewidth = 3)
#plt.plot(Epsilon, PGD_GD, 'k--', label = 'PGD attack', linewidth = 3)
#plt.plot(Epsilon, PGD_CW, '#800000', label = 'CW attack', linewidth = 3, linestyle = '--')
#plt.plot(Epsilon, PGD_FGSM, '#0082c8', label = 'FGSM attack', linewidth = 3, linestyle = '--')
#plt.xlabel(r'$\mathbf{\epsilon}$', fontsize = 30, weight = 'bold')
#plt.ylabel('Zero-one-loss', fontsize = 25, weight = 'bold')
#plt.legend(loc = 'upper center', fontsize = 15)
#plt.title("PGD adversarial training", fontsize = 20, weight = 'bold')
#plt.savefig('PGD-plot.png', format = 'png')

plt.clf()

# Plotting on one axis for SDP network
plt.plot(Epsilon, SDP_SPE, '#911eb4', label = 'Spectral', linewidth = 3)
plt.plot(Epsilon, SDP_FRO, 'c', label = 'Frobenius', linewidth = 3)
plt.plot(Epsilon, SDP_SDP, '#f58231', label = 'Proposed SDP', linewidth = 3)
plt.plot(Epsilon, SDP_GD, 'k--', label = 'PGD attack', linewidth = 3)
plt.plot(Epsilon, SDP_CW, '#800000', label = 'CW attack', linewidth = 3, linestyle = '--')
plt.plot(Epsilon, SDP_FGSM, '#0082c8', label = 'FGSM attack', linewidth = 3, linestyle = '--')
plt.xlabel(r'$\mathbf{\epsilon}$', fontsize = 30, weight = 'bold')
plt.ylabel('Zero-one-loss', fontsize = 25, weight = 'bold')
plt.legend(loc = 'upper center', fontsize = 15)
plt.title("Proposed SDP training", fontsize = 20, weight = 'bold')
plt.savefig('SDP-plot.png', format = 'png')

plt.clf()

# Comparing AT-NN and SDP-NN against attacks

plt.plot(Epsilon, SDP_GD, 'b--', label = 'PGD on SDP-NN', linewidth = 3)
plt.plot(Epsilon, PGD_GD, 'r--', label = 'PGD on AT-NN', linewidth = 3)
plt.legend(loc = 'upper center', fontsize = 15)
plt.xlabel(r'$\mathbf{\epsilon}$', fontsize = 30, weight = 'bold')
plt.ylabel('Zero-one-loss', fontsize = 25, weight = 'bold')
plt.title("Same attack on different networks", fontsize = 20, weight = 'bold')
plt.savefig('SDP-PGD.png', format = 'png')
