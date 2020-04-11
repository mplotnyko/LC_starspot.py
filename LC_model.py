import numpy as np
import mpmath as mp
from scipy.optimize import curve_fit
class transit_fit(object):
    """
    Fitting light curve data for a transit given limb darkening model,
    the default is "uniform" model where no limb darkening is used.
    Use method="quad_ld" to use quadratic limb darkening model (2 coef) or use
    method="nliner_ld" (4 coef) for non linear model.
    Input: the time and flux data, if the model parametars are known before pass
    them as an array of [ap_rs,rp_rs,inc,period]
    Ratio of semi-major axis to radius of star (ap_rs),
    ratio of planet to star radius (rp_rs), inclanation (inc), 
    period of transit (period). 
    Limb darkening (ld_coef) optional depending on method chosen.
    """
    def __init__(self,theta=None,ld_coef=None,method="uniform"):
        if method=="uniform":
            self.model = self.flux_uni
        elif method=="quad_ld":
            if ld_coef is not None:
                if len(ld_coef)!=2:
                    print("something wrong with ld_coef")
                    return
                gamma_1,gamma_2 = ld_coef
                c0 = 1-gamma_1-gamma_2
                self.ld_coef = [c0,0,gamma_1+2*gamma_2,0,-gamma_2]
                self.n = np.arange(5)
                self.omega = np.sum(self.ld_coef/(self.n+4))
            self.model = self.flux_quadld
        elif method=="nlinear_ld":
            if ld_coef is not None:
                if len(ld_coef)!=4:
                    print("something wrong with ld_coef")
                    return
                self.ld_coef = np.array([1-sum(ld_coef),*ld_coef])
                self.n = np.arange(5)
                self.omega = np.sum(self.ld_coef/(self.n+4))
            self.model = self.flux_nonlinld
        else:
            print("Wrong method specified, choose from the following:\n",
                  "    uniform, quad_ld, nlinear_ld")
            return
        if theta is not None:
            ap_rs,rp_rs,inc,period = theta
            if ap_rs<=0 or rp_rs<=0 or inc<=0 or period<=0:
                print("Something wrong with the system parameters\n",
                      "    z0,p,i,T = ",ap_rs,rp_rs,inc,period)
                return 
            self.z0 = ap_rs
            self.p = rp_rs
            self.i = inc
            self.period = period
        else: 
            self.p=-0.1
    def ls_fit(self,time,flux,Te=None):
        # do ls_fit
        if self.ld_coef is None:
            ld_coef = [1,0,0,0,0]
        else:
            ld_coef = self.ld_coef  
        if self.p>0.:
            args = [self.z0,self.p,self.i,self.period,0.1,0.1]
        else:
            period = 100 #maybe use star mass for better aproximation
            inc = np.pi/2
            z0 = 100
            p = 0.1
            args = [z0,p,inc,period,0.1,0.1]
            
        if Te is None:
            period_ulim = 365*100
            period_llim = 0
        else:
            # if period is know correctly.
            period_ulim = self.period+Te
            period_llim = self.period-Te
            
        popt,pocv = curve_fit(self.LSModel_QuadLD,time,flux,
                              p0 = args,
                              bounds=([0,0,0,period_llim,0,0], [400,1,np.pi/2+0.1,period_ulim,1,1]),
                              maxfev=10000)
        return popt,pocv 
    def LSModel_QuadLD(self,time,ap_rs,rp_rs,inc,period,gamma_1,gamma_2):
        c0 = 1-gamma_1-gamma_2
        self.ld_coef = [c0,0,gamma_1+2*gamma_2,0,-gamma_2]
        self.n = np.arange(5)
        self.omega = np.sum(self.ld_coef/(self.n+4))
        self.z0 = ap_rs
        self.p = rp_rs
        self.i = inc
        self.period = period
        return self.transit_model(time)
    
    def transit_model(self,time):
        """
        Transit model for analytical light curve, given transit time data.
        """
        z_t = self.get_z(time)
        p = self.p
        Flux = np.empty(len(time))
        for i,z in enumerate(z_t): 
            if (p>0 and 1+p<=z) or (0<=z and p==0):
                Flux[i] = 1.
            else:
                Flux[i] = self.model(p,z)
        return Flux
    def flux_uni(self,p,z):
        return 1 - self.lambda_e(p,z)
    def flux_quadld(self,p,z):
        c0,c1,c2,c3,c4  = self.ld_coef
        lambda_d,eta_d = self.cases(p,z)
        if z<p:
            lambda_d +=2/3    
        Flux = 1 - ( (1-c2)*self.lambda_e(p,z) + 
                              c2*( lambda_d ) - c4*eta_d )/4/self.omega
        return Flux
    def flux_nonlinld(self,p,z):
        return self.case_nonld(p,z)
    def get_z(self,t):
        w = 2*np.pi/self.period
        return self.z0*np.sqrt(np.sin(w*t)**2+np.cos(self.i)**2*np.cos(w*t)**2)
    def cases(self,p,z):
        if (p>0 and 1+p<=z) or (0<=z and p==0):
            return 0.,0.
        elif p>0 and 1/2+abs(p-1/2)<z<1+p:
            return self.lambda_i(1,p,z),self.eta_1(p,z)
        elif 0<p<1/2 and p<z<1-p:
            return self.lambda_i(2,p,z),self.eta_2(p,z)
        elif 0<p<1/2 and abs(z+p-1)<1e-5:
            return self.lambda_i(5,p,z),self.eta_2(p,z)
        elif 0<p<1/2 and z==p:
            return self.lambda_i(4,p,z),self.eta_2(p,z)
        elif p==1/2 and z==1/2:
            return 1/3-4/9/np.pi,3/32
        elif p>1/2 and z==p:
            return self.lambda_i(3,p,z),self.eta_1(p,z)
        elif p>1/2 and abs(1-p)<=z<p: 
            return self.lambda_i(1,p,z),self.eta_1(p,z)
        elif 0<p<1 and 0<z<1/2-abs(p-1/2):  
            return self.lambda_i(2,p,z),self.eta_2(p,z)
        elif 0<p<1 and z==0:
            return self.lambda_i(6,p,z),self.eta_2(p,z)
        elif p>1 and 0<=z<p-1:
            return 1.,1.
    def case_nonld(self,p,z):
        ld_coef = self.ld_coef 
        n = self.n 
        omega = self.omega
        if p>0 and 1/2+abs(p-1/2)<z<1+p:
            return 1 - np.sum( self.N(p,z)*ld_coef/(n+4))/2/np.pi/omega
        elif 0<p<1/2 and (p<z<1-p or z==1-p):
            L = p**2*(1-p**2/2-z**2)
            return 1 - (ld_coef[0]*p**2 + ld_coef[4]*L +
                            2*np.sum(self.M(p,z,n[1:4])*ld_coef[1:4]/(n[1:4]+4)) 
                        )/4/omega
        elif 0<p<1/2 and z==p:
            hyp = np.zeros(len(n))
            for i in range(len(n)):
                hyp[i] = self.hyp2f1(1/2,-n[i]/4-1,1,4*p**2)
            return 1/2 + np.sum( ld_coef/(n+4)*hyp )/2/omega
        elif p==1/2 and z==1/2:
            coef = np.zeros(len(n))
            for i in range(len(n)):
                coef[i] = self.gamma(1.5+n[i]/4)/self.gamma(2+n[i]/4)
            return 1/2 + np.sum( ld_coef/(n+4)*coef )/2/np.sqrt(np.pi)/omega
        elif p>1/2 and z==p:
            coef = np.zeros(len(n))
            for i in range(len(n)):
                coef[i] = (mp.beta(0.5,n[i]/4+2) *
                           self.hyp2f1(0.5,0.5,5/2+n[i]/4,1/4/p**2))
            return 1/2 + np.sum( ld_coef/(n+4)*coef )/4/p/np.pi/omega
        elif p>1/2 and abs(1-p)<=z<p: 
            return  - np.sum( self.N(p,z)*ld_coef/(n+4))/2/np.pi/omega
        elif 0<p<1 and 0<z<1/2-abs(p-1/2):  
            L = p**2*(1-p**2/2-z**2)
            return (ld_coef[0]*(1-p**2)+ld_coef[4]*(0.5-L) -
                        2*np.sum(self.M(p,z,n[1:4])*ld_coef[1:4]/(n[1:4]+4))
                        )/4/omega
        elif 0<p<1 and z==0:
            return np.sum(ld_coef*(1-p**2)**(n/4+1)/(n+4))/omega
        elif p>1 and 0<=z<p-1:
            return 0.
    def lambda_e(self,p,z):
        if 1+p<z:
            return 0.
        elif 1-p<z<=1+p:
            k0 = np.arccos( (p**2 + z**2 -1)/2/p/z )
            k1 = np.arccos( (1 - p**2 + z**2)/2/z ) 
            return 1/np.pi * ( p**2*k0 + k1 - np.sqrt(z**2 -
                                                      ( 1+z**2-p**2 )**2/4 ) )
        elif z<=1-p:
            return p**2
        elif z<=p-1:
            return 1.
    def lambda_i(self,i,p,z):
        a = (z-p)**2
        b = (z+p)**2
        if z!=0:
            k = np.sqrt((1-a)/4/z/p)
        q = p**2-z**2
        if i==1:
            return ( 1/9/np.pi/np.sqrt(p*z) * 
                        (((1-b)*(2*b+a-3)-3*q*(b-2) )*self.ellip_fir(k) 
                        + 4*p*z*(z**2+7*p**2-4)*self.ellip_sec(k) 
                             - 3*q/a*self.ellip_thi((a-1)/a,k) )
                    )
        elif i==2:
            return ( 2/9/np.pi/np.sqrt(1-a)* 
                    ((1-5*z**2+p**2+q**2 )*self.ellip_fir(1/k) 
                        + (1-a)*(z**2+7*p**2-4)*self.ellip_sec(1/k) 
                         - 3*q/a*self.ellip_thi((a-b)/a,1/k))
                   )
        elif i==3:
            return ( 1/3+16*p/9/np.pi*(2*p**2-1)*self.ellip_sec(1/2/k) - 
                        (1-4*p**2)*(3-8*p**2)/9/np.pi/p*self.ellip_fir(1/2/k) )
        elif i==4:
            return 1/3 + 2/9/np.pi*( 4*(2*p**2-1)*self.ellip_sec(2*k)
                                        + (1-4*p**2)*self.ellip_fir(2*k))  
        elif i==5:
            return 2/3/np.pi*np.arccos(1-2*p)-4/9/np.pi*(3+2*p-8*p**2)
        elif i==6:
            return -2/3*(1-p**2)**1.5
    def N(self,p,z):
        a = (z-p)**2
        b = (z+p)**2
        N = len(self.n)
        res = np.zeros(N)
        for n in range(N):
            beta = mp.beta((n+8)/4,1/2)
            appell = self.appellf1(1/2,1,1/2,(n+10)/4,(a-1)/a,(1-a)/(b-a))
            gauss = self.hyp2f1(1/2,1/2,(n+10)/4,(1-a)/(b-a))
            res[n] = (1-a)**(n/4+3/2)/np.sqrt(b - a)*beta*(
                              (z**2-p**2)/a*appell - gauss) 
        return res
    def M(self,p,z,N):
        a = (z-p)**2
        b = (z+p)**2
        res = np.zeros(len(N))
        for i,n in enumerate(N):
            appell = self.appellf1(1/2,-n-1,1,1,(b-a)/(1-a),(a-b)/a)
            gauss = self.hyp2f1(-n-1,1/2,1,(b-a)/(1-a))
            res[i] = (1-a)**(n/4+1)*((z**2-p**2)/a*appell - gauss)
        return res
    def hyp2f1(self,a,b,c,d):
        return mp.hyp2f1(a,b,c,d,maxterms=10000)
    def appellf1(self,a,b1,b2,c,x,y):
        if abs(x)<1 and abs(y)<1:
            return mp.appellf1(a,b1,b2,c,x,y,maxterms=10000)
        else:
            if y>0.9899:
                y = 0.989999
            else:
                y=0.01
            return abs(mp.appellf1(a,b1,b2,c,x,y,maxterms=10000))
    def gamma(self,a):
        return mp.gamma(a)
    def eta_1(self,p,z):
        a = (z-p)**2
        b = (z+p)**2
        k0 = np.arccos( (p**2 + z**2 -1)/2/p/z )
        k1 = np.arccos( (1 - p**2 + z**2)/2/z ) 
        return ( 1/2/np.pi*(k1+2*self.eta_2(p,z)*k0 - 
                          (1+5*p**2+z**2)*np.sqrt((1-a)*(b-1))/4) )
    def eta_2(self,p,z):
        return p**4/2 + p**2*z**2 
    def ellip_fir(self,k):
        return mp.ellipk(k**2)
    def ellip_sec(self,k):
        return mp.ellipe(k**2)
    def ellip_thi(self,n,k):
        return mp.ellippi(n,np.pi/2,k**2)