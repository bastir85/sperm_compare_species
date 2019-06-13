import h5py
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter as GaussFilter

from .curvature import compute_curvature_xyz
from .spectrogram import spectrogram2
spacing=11/20

def load_filament_raw_h5(H5FP):
    sizex, sizey, frames = H5FP["info/sizeInfo"]
    def get_frame(t):
        x = H5FP["arcLengthPoints/x/{}".format(t)].value*spacing
        y = -H5FP["arcLengthPoints/y/{}".format(t)].value*spacing
        z = np.zeros_like(x)    
        return x, y, z
    
    raw_data = np.array([get_frame(t) for t in range(1,frames+1)])
    num_points =  raw_data.shape[-1]
    T = np.arange(frames)/500
    index = pd.MultiIndex.from_product([T, range(num_points)],
                                   names=["time", "s"],)
    filament = pd.DataFrame(raw_data.swapaxes(1,2).reshape(-1,3),
             columns=["X", "Y", "Z"],  index=index)
    filament.loc[filament.X == 0.0] = None
    
    
    Xn = (H5FP['preprocess/cuttingCenterX'].value - sizex/2.)*spacing
    Yn = -(H5FP['preprocess/cuttingCenterY'].value - sizey/2.)*spacing
    X = (H5FP['head/x'].value)*spacing
    Y = -(H5FP['head/y'].value)*spacing
    phi = H5FP['head/phi'].value/180.*np.pi   
    head = pd.DataFrame({'Xn': Xn, 'Yn': Yn, 'X': X, 'Y': Y, 'phi': phi},
                        index=T)
    p0 = filament.xs(0, level=1)[["X","Y"]]
    d = p0 - head[["X","Y"]] 
    head["phi2"] = np.unwrap(np.arctan2(d.Y, d.X))
    
             
    head.loc[head.X == 0.0] = None
    
    return filament, head



def average_power(freqs, powers):
    w0_idxs = powers.T.argmax(axis=1)
    average_power = []
    for idx, p in enumerate(powers.T):
        w0 = freqs[w0_idxs[idx]]
        average_power.append((freqs/w0, p))
    return np.array(average_power).swapaxes(1,2)

def analyse_shape(head, NFFT=250, pca=False, smoothing=200):
    if smoothing:
        G = lambda x: GaussFilter(x, smoothing)
    else:
        G = lambda x: x

    if pca:
        kappa = head.kappa_s
    else:
        kappa = head.kappa
    framerate = np.diff(kappa.index).mean()
    if any(np.isnan(kappa)):
        print("Some frames are missing! Will create errors!")
    # Full FFT to get taub
    F = np.fft.rfft(kappa)
    fr = np.fft.rfftfreq(len(kappa), d=1/framerate)
    
    fr_m = fr[np.abs(F).argmax()]
    taub = 1/fr_m
    ## moving fourier analysis of the beat pattern
    powers, phangle, freqs, times, x , res = spectrogram2(kappa.values,
                                                         NFFT, 1./framerate, identify_peaks=True,
                                                         noverlap=NFFT-1)
    CM,C0,C1,C2,W,phi = res 
    spectrum = average_power(freqs, powers)

    mask = W==0.0
    if any(mask):
        print("Sperm got stuck!")

    W[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), W[~mask])
    C0[mask] = 0.0
    C0[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), C0[~mask])
    C1[mask] = 0.0
    C1[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), C1[~mask])
    C2[mask] = 0.0
    C2[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), C2[~mask])
    phi[mask] = 0.0
    phi[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), phi[~mask])
    k40_shape = pd.DataFrame({"time":times, "omega":W, "C0":C0, 
        "C1":C1,"C2":C2, "phase":phi, "stuck": mask}
                          ).set_index("time")

    head["phi_vel"] = head.phi.diff(2).shift(-1).fillna(0.0)/2/framerate/2/np.pi
    head["phi_vel_s"] = G(head.phi_vel) 
    
    res = pd.merge(k40_shape, head, left_index=True, right_index=True, 
                   how="outer").fillna(method="ffill")
    res["phi_vel_normed"] = res.phi_vel_s / G(res.omega)
    res["phi_vel_normed_s"] = res.phi_vel_normed
    
    from scipy.optimize import minimize
    
    
    def R(ph0=0.0):
        return -correlate(res2.phi_vel_normed_s, G(res2.C1*np.sin(res2.phase + ph0))).loc[0]
    
    res2 = res.dropna()
    result = minimize(R, [0]) 
    ph0 = result.x[0] % (2*np.pi)
    res["phase_eff"] = res.phase+ph0
    res["C2_SIN"] = G(res.C1*np.sin(res.phase_eff))
    res["kappa_mean_s"] = G(res.kappa_mean)
    
    return res, spectrum 


def correlate(a, b):
    c0f = (np.correlate(np.ones_like(a), np.ones_like(b), mode="full")-1)*np.sqrt((a**2).mean()*(b**2).mean())
    conv  = pd.Series(np.correlate(a, b,
                      mode='full')/c0f, 
                      index=np.arange(-len(a)+1,len(a)))
    return conv 


def get_eigenmodes(bb, thres=0.05):
    'Calculate eigenmodes for bb[t,s]'
    M = np.einsum("ij,il->jl",bb,bb)
    eig, d = np.linalg.eigh(M)
    eig = np.sqrt(eig[::-1])
    eig = eig / np.nansum(eig)
    num_ev = np.cumsum(eig> thres).max()
    A = d[:,:-1-num_ev:-1]
    res = np.dot(bb,A)
    return res, A, eig,num_ev


def plot_curvature(k):
    sel = k.unstack()
    T = sel.index
    S = sel.columns*spacing
    Z = sel.values
    Ts = np.vstack([T]*len(S)).T
    Ss = np.vstack([S]*len(T))
    plt.pcolor(Ts, Ss, Z, cmap=plt.cm.viridis, vmin=-0.5, vmax=.5)
    plt.colorbar(label="curvature $[\mu m^-1]$")
    plt.xlabel("t [s]")
    plt.ylabel(r"s $[\mu m]$")


def get_eigenmodes(bb, thres=0.05):
    'Calculate eigenmodes for bb[t,s]'
    M = np.einsum("ij,il->jl",bb,bb)
    eig, d = np.linalg.eigh(M)
    eig = np.sqrt(eig[::-1])
    eig = eig / np.nansum(eig)
    num_ev = np.cumsum(eig> thres).max()
    A = d[:,:-1-num_ev:-1]
    res = np.dot(bb,A)
    
    return res, A, eig,num_ev

def smooth_eigen(curv):
    res, A, eig, num_ev = get_eigenmodes(curv)
    print(num_ev)
    return np.dot(res,A.T)

    

