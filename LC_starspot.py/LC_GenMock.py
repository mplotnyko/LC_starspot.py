import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import LC_model
def generate_mock_data(theta,ld_coef,transit_window,spot_location, spot_strength, spot_width):
    KEPLER_NOISE_LEVEL = 200 #ppm
    KEPLER_SAMPLE_RATE = 1765.45 #sec
    
    time_array = np.arange(0 - transit_window / 24.0, 
                           0 + transit_window / 24.0, 
                           1.0 / 60.0 / 24.0)
    LC = LC_model.transit_fit(theta,ld_coef,method="quad_ld")
    transit_light_curve = LC.transit_model(time_array)
    x_true = time_array - np.median(time_array)
    y_true = (transit_light_curve-1)*1e6 #ppm
    num_samples = int((x_true[-1]-x_true[0])*24*3600/KEPLER_SAMPLE_RATE)
    indexes = np.linspace(0,x_true.size-1,num_samples).astype(int)
    x_mock = x_true[indexes]
    anomaly = stats.norm.pdf(x_true, spot_location, spot_width)
    anomaly /= np.max(anomaly)
    anomaly *= spot_strength
    y_anomaly = y_true + anomaly

    y_mock = y_anomaly[indexes] + np.random.normal(0,KEPLER_NOISE_LEVEL,size=x_mock.size)
    res = [x_mock, y_mock, x_true, y_true, anomaly, y_anomaly]
    return res
def plot_mock_data(x_mock, y_mock, x_true, 
                   y_true, anomaly, y_anomaly,name=None,res=False,
                   label='Underlying Transit Light Curve', **kwargs):
    r = 1
    h = [1]
    if res:
        r = 2
        h = [3, 1]
    fig,axs = plt.subplots(r,1,figsize=(10,7), 
                           gridspec_kw={'height_ratios': h, 'hspace':0})
    t = np.linspace(min(x_true),max(x_true),len(anomaly))
    if res:
        ax = axs[0]
    else:
        ax = axs
    ax.plot(t, 
             anomaly, 
             color = 'blue', 
             label='Perturbation')
    ax.plot(x_true, 
             y_true, 
             color = 'red', 
             label=label)
    ax.plot(t, 
             y_anomaly, 
             color = 'blue', 
             linestyle = '--', 
             label = 'Perturbed Transit Light Curve')
    ax.plot(x_mock, 
             y_mock, 
             'x', 
             color = 'black', 
             label = 'Mock Data')
    ax.legend()
    ax.set_ylabel("Flux Drop Fraction [ppm]", size=14)
    ax.set_title("Example Mock Data",size=18)
    ax.grid(alpha=0.5)
    if name is not None:
        plt.savefig(name,kwargs)
    if res:
        ax.set_xticklabels([])
        ax = axs[1]
        ax.plot(t, 
                 np.zeros(len(t)), 
                 color = 'blue', 
                 linestyle = '--')
        ax.plot(x_true, 
                 y_mock-y_true, 
                 'x', 
                 color = 'red', 
                 label = 'Mock Data')
        ax.set_ylabel("Residual",size=14)
        ax.grid(alpha=0.5)
        ax.set_ylim(min(y_mock-y_true)-200,2000+200)
    ax.set_xlabel("Time [Days]", size=14)
    return fig