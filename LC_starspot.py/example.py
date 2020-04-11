from LC_GenMock import generate_mock_data,plot_mock_data
import LC_model
import numpy as np
import emcee

#Starspot parameters
transit_window = 10 #hours
spot_stength = 2000 #ppm
spot_width = 0.03 #days
spot_location = transit_window/24*0.5 #days
starspot = [transit_window,spot_location,spot_stength,spot_width]

#Transit parameters
theta_0 = [140.5,0.0576,1.571, 224.22] 
ld_coef = [0.47,0.19]
results = generate_mock_data(theta_0,ld_coef,transit_window,*starspot)
fig = plot_mock_data( *results) #plotting

#Least-suqares estimate
theta = np.array([140,0.1,np.pi/2,224,0.5,0.5])
time = x_mock
flux = y_mock*1e-6+1

fit = LC_model.transit_fit(theta[:-2],theta[-2:],method="quad_ld")
popt,pocv = fit.ls_fit(time,flux,Te=1)
print("Best estimate LS:\n", popt)

#MCMC estimate
def lnprior(theta,theta_m,theta_lim):
    tt=0
    for i in range(4):
        if abs(theta_m[i]-theta[i])<theta_lim[i]*5:
            tt +=1
    if theta[4]+theta[5]<1 and (theta>0).all() and tt==4: 
        return 0.0
    return -np.inf
def lnlike(theta,time,flux,sig):
    fit = LC_model.transit_fit(theta[:-2],theta[-2:],method="quad_ld")
    model = fit.transit_model(time)
    return -np.sum((model-flux)**2/sig**2)  #+ np.log(sig))
def lnprob(theta,args):
    time,flux,sig,theta_m,theta_lim = args
    lp = lnprior(theta,theta_m,theta_lim)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,time,flux,sig)
#initialize
theta_m = popt
theta_lim = [3,0.02,0.002,0.1,0.1,1e-10]
sig1 = np.zeros(len(time))+1e-4
ndim, nwalkers = len(theta), 100
iteration = 2000
pos = np.zeros([ndim,nwalkers])

for i in range(ndim):
    pos[i] = np.random.normal(theta_m[i],theta_lim[i],nwalkers)
args = [time,flux,sig1,theta_m,theta_lim]

#sample using emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,args=[args],threads=8)
pos = sampler.run_mcmc(pos.T, iteration);
np.save('chain.dat',chain) #used after the simulation
