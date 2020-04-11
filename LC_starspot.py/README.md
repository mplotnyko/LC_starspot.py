# Transit Light curve with a Starspot in Python

Python implementation of modelling transit light-curve with the addition of a Starspot.
The analytical model follows the work of [Mandel & Agol (2002)](https://ui.adsabs.harvard.edu/abs/2002ApJ...580L.171M), with an option to compute the model using:
* Uniform source.
* Quadratic Limb darkening.
* Non-linear Limb darkening.
In the following code, the starspot is generated using a Gaussian perturbation and one can perform the best fit model on the mock data.
## Installation

1. Clone the repository

```git clone https://github.com/fhorrobin/LC_starspot.py```

## Example 
This example can be found in the example.py file.
'''
from LC_GenMock import generate_mock_data, plot_mock_data
import LC_model
import numpy as np
'''
The first step is to generate transit data with a Starspot.
Then the duration of transit and the strength of the perturbation due to the Starspot needs to be specified.
'''
transit_window = 10 #hours
spot_stength = 2000 #ppm
spot_width = 0.03 #days
spot_location = transit_window/24*0.5 #days
starspot = [transit_window,spot_location,spot_stength,spot_width]
'''
Using quadratic limb darkening model the transit parameters (theta_0 & ld_coef) can be specified to generate the data.
'''
theta_0 = [140.5,0.0576,1.571,224.22] 
ld_coef = [0.47,0.19]
results = generate_mock_data(theta_0,ld_coef,transit_window,*starspot)
'''
These results can be easily ploted using the following command:
'''
fig = plot_mock_data( *results)
'''
If one wishes then to obtain a best fit results for the generated mock data, here is an example of using [emcee](https://github.com/dfm/emcee).
First get an estimate of the parameters using least-squares method:
'''
theta = np.array([140,0.1,np.pi/2,224,0.5,0.5])
time = x_mock
flux = y_mock*1e-6+1

fit = LC_model.transit_fit(theta[:-2],theta[-2:],method="quad_ld")
#assunme 1 day error in period
popt,pocv = fit.ls_fit(time,flux,Te=1) 
print("Best estimate LS:\n", popt)
'''
Then a log-likelihood and posterior can be set up as following:
'''
import emcee
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
    return -np.sum((model-flux)**2/sig**2)  
def lnprob(theta,args):
    time,flux,sig,theta_m,theta_lim = args
    lp = lnprior(theta,theta_m,theta_lim)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,time,flux,sig)
'''
Then setting the simulation to have 100 walkers with 2000 steps, the sampler is initialized
'''
theta_m = popt
theta_lim = [3,0.02,0.002,0.1,0.1,1e-10]
sig1 = np.zeros(len(time))+1e-4
ndim, nwalkers = len(theta), 100
iteration = 2000

pos = np.zeros([ndim,nwalkers])
for i in range(ndim):
    pos[i] = np.random.normal(theta_m[i],theta_lim[i],nwalkers)
args = [time,flux,sig1,theta_m,theta_lim]
'''
After that the MCMC simulation can be started:
'''
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,args=[args],threads=8)
pos = sampler.run_mcmc(pos.T, iteration);
'''
The results of the chains are stored in sampler, and can be acessted easily by
'''
samples = sampler.get_chain()
'''