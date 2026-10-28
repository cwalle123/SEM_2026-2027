import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import math, time

# C1,A1,B2,A2,C3:   #for Zernike-Nijboer, see for example Born and Wolf, Principles of Optics
def ZernikePhase(a:float, b:float, theta:float, aber:np.ndarray):  #theta = half opening angle
    C1  = aber[0]
    A1r = aber[1];    A1i = aber[2]
    B2r = aber[3];    B2i = aber[4]
    A2r = aber[5];    A2i = aber[6]
    C3  = aber[7]
    A3r = aber[8];    A3i = aber[9]
    a2 = a**2;   b2 = b**2;   th2 = theta**2
    S  = 0.5*C1*(a2 + b2) + 0.5*A1r*(a2 - b2) +  A1i*a*b
    S += (B2r*a - B2i*b)*(a2 + b2 - (2/3)*th2) + A2r*a*((a2/3) - b2) + A2i*b*(a2 - (b2/3))
    S += 0.25*C3*(a2 + b2)*(a2 + b2 - th2)
    S += A3r * (a**4 + b**4 - 6*a**2*b**2) + A3i * (4*a*b*(a**2 - b**2)) # Four-Fold Astigmatism
    return S

def fill_aperture(i:int, j:int, xc: int, zpf:float, abert:np.ndarray, thetat:float):
    return np.where((i-xc)**2 + (j-xc)**2 > thetat**2, 0j, np.exp((1j)*ZernikePhase(j-xc,i-xc,thetat,abert)))

#aber: C1, A1r, A1i, B2r, B2i, A2r, A2i, C3, a3r, a3i
def calcPSFOTFwave(n:int, zpf:float, lamb:float, theta:float, d50geo:float, aber:np.ndarray):  #zpf = zero pad factor, theta = half opening angle
    assert(zpf>=1) 
    k = 2*math.pi/lamb            #lamb = wavelength
    pix = math.pi/(k*theta*zpf)   #pix = PSF pixel size
    sigxt = d50geo/(pix*math.sqrt(math.log(256.0)))   # variables with last letter = t --> dimensionless parameter
    xc = n//2
    tht = 0.5*n/zpf   
    thr = theta/tht   # thr = theta-ratio = properly scaled theta
    abt = np.copy(aber)   
    abt[0] *= k*thr**2;    abt[1] *= k*thr**2;    abt[2] *= k*thr**2
    abt[3] *= k*thr**3;    abt[4] *= k*thr**3;    abt[5] *= k*thr**3;    abt[6] *= k*thr**3
    abt[7] *= k*thr**4;    abt[8] *= k*thr**4;    abt[9] *= k*thr**4
    i,j = np.indices((n,n))                      # vectorized -->
    ap = fill_aperture(i,j,xc,zpf,abt,tht)     # fast
    
    psi = (1/(math.sqrt(math.pi)*tht*n))*np.fft.fft2(ap)   
    psf = (abs(psi))**2
    otf = np.fft.fft2(psf)
    assert(otf[0,0] != 0)
    otf /= abs(otf[0,0])
    #next, blur psf with Gaussian:  (vectorized --> fast)
    if (d50geo>0):
        sigkt = n/(2*math.pi*sigxt)
        X,Y = np.indices((n,n))
        gaus = np.exp(-0.5*((X-xc)**2 + (Y-xc)**2)/sigkt**2)   #xc = n//2
        gaus = np.fft.fftshift(gaus)
        otf = otf*gaus
        psf = abs(np.fft.ifft2(otf))   #normalized: np.sum(psf) ~ 1
    return (psf/np.max(psf), otf)      #both: center in (0,0)


# -------------------------- begin program -----------------------------------------------------------------------------------
# simulated wave-optical PSF, excluding Cc.  See for example Hawkes and Kasper, volume III
npsf = 1024     #PSF: npsf x npsf pixels          
theta = 0.015    #half opening angle [rad] 
d50geo = 1e-9   #[meter] (e-9)
c1 = 0.5e-6    #defocus [meter] (e-6)
a1r = 0;      a1i = 0;   #astigmatism (e-8)
b2r = 0;      b2i = 0;  #coma (e-6)
a2r = 0;      a2i = 10e-6;  #3-fold astigmatism (e-6)
a3r = 0;      a3i = 0;  #4-fold astigmatism (e-4)
c3 = 0             #Cs (e-3)
U = 30000;  
zpf = 5  #zero pad factor, >1 and <~ 15

aber = (c1,a1r,a1i,b2r,b2i,a2r,a2i,c3,a3r,a3i)
aber = np.array(aber)
eps = 9.785e-7
Ur = U + eps*(U**2)
lamb = 1.226e-9/Ur**0.5   #electron wavelength
pixsize = 0.5*lamb/(zpf*theta)
print(f"PSF: pixel size = {(pixsize*1e9):.3f}  nm")

t1 = time.perf_counter()
(psf, _) = calcPSFOTFwave(npsf, zpf, lamb, theta, d50geo, aber)   # or (psf, otf) = ...
t2 = time.perf_counter()
print(f"calculation time  = {(t2-t1):.3f} sec.")
psf = np.abs(np.fft.fftshift(psf))  #centered
plt.imshow(psf, cmap="gray");   plt.show()
#np.save("psf.npy",psf) 

'''
#cross-sections:
hn = npsf//2
PSFx = psf[hn:1+hn, 0:npsf].T     #PSFx = blue
PSFy = psf[0:npsf, hn:1+hn]       #PSFy = red
plt.plot(PSFx,'b-',PSFy,'r-');   plt.show()
'''