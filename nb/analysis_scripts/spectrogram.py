# -*- coding: utf-8 -*-
"""
Modified: 27.08.1

@author: Guglielmo Saggiorato <g.saggiorato@fz-juelich.de> or <astyonax@gmail.com>
@author: Sebastian Rode - python3 update
"""

import numpy as np
from scipy import signal

def windfft(x,window=signal.blackmanharris):
    L=x.shape[0]
    M=window(L)
    return np.fft.rfft(x*M)/M.sum()*2

def wind(x,NFFT,noverlap=0):
    for j in range(0,len(x),NFFT-noverlap):
        if NFFT+j<len(x):
            yield x[j:NFFT+j]
        else:
            return

def spectrogram2(x,NFFT,Fs,window=signal.blackmanharris,noverlap=0,identify_peaks=False):
    spectrogram=[]
    phangle=[]
    pw=lambda x:np.abs(x)**2
    phasew=lambda x:np.angle(x)

    for i,q in enumerate(wind(x,NFFT,noverlap)):
        dfft=windfft(q,window=window)
        spectrogram.append(pw(dfft))
        phangle.append(phasew(dfft))

    spectrogram=np.array(spectrogram).T
    phangle=np.array(phangle).T
    freqs=np.fft.rfftfreq(NFFT,1./Fs)

    DT=NFFT-noverlap

    times=np.arange(0.,len(spectrogram[0]))/Fs*DT + NFFT/2/Fs 
    if not identify_peaks:
        return spectrogram,phangle,freqs,times

    x=np.argmax(spectrogram,axis=0)
    x[x>spectrogram.shape[0]/3]=0
    CM=spectrogram[:,0]

    LL=range(spectrogram.shape[1])
    C0=np.array([spectrogram[x[i],i] for i in LL])**.5
    C1=np.array([np.sum([spectrogram[2*x[i],i], spectrogram[2*x[i]+1,i], 
                         spectrogram[2*x[i]-1,i]]) for i in LL])**.5
    C2=np.array([spectrogram[3*x[i],i] for i in LL])**.5
    PHI=np.array([phangle[2*x[i],i]-phangle[x[i],i]*2 for i in LL])
    W=freqs[x]

    return spectrogram,phangle,freqs,times,x.astype(float),[CM,C0,C1,C2,W,PHI]
