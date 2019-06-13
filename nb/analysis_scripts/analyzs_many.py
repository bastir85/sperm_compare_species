from scipy.ndimage.filters import gaussian_filter as GaussFilter
import pandas as pd
import numpy as np

from .curvature import compute_curvature_xyz
from .utils import convert_xyz_df_coord_index_to_column, load_xyz
from . import second_harm_tools

import h5py

## Load Bull
def get_bull(path, rel_s=2/3, t_range=None):
    head = np.loadtxt(path + ".rawheadAngleXY", delimiter=';')
    data = np.dstack([np.loadtxt(path + ".rawtailX", delimiter=';'),
                      np.loadtxt(path + ".rawtailY", delimiter=';')])
    if t_range:
        data = data[t_range]
        head = head[t_range]

    head = pd.DataFrame(head, columns=["phi","X","Y"])
    head["phi"] = np.unwrap(head.phi / 180*np.pi)

    k = compute_curvature_xyz(xyz=data)
    ks_s = second_harm_tools.smooth_eigen(k)

    s0 = int(round((len(k.T)-1)*rel_s))

    head["kappa_mean"] = k.mean(axis=1)
    head["kappa"] = k.T[s0]
    head["kappa_s"] = ks_s.T[s0]

    head.index = head.index/200
    return head, k, data

## Load Sea Urchin
def _get_sea_urchin(H5FP, rel_s=2/3, t_range=None):
    spacing=11/20
    sizex, sizey, frames = H5FP["info/sizeInfo"]
    def get_frame(t):
        x = H5FP["arcLengthPoints/x/{}".format(t)].value*spacing
        y = -H5FP["arcLengthPoints/y/{}".format(t)].value*spacing
        z = np.zeros_like(x)    
        return x[:75], y[:75], z[:75]
    
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
    
    filament["k"] = np.nan
    for t, frame in list(filament.groupby("time")):
        k2 =compute_curvature_xyz(x=frame.X.values[np.newaxis], y=frame.Y.values[np.newaxis])
        filament.loc[t].k = pd.Series(k2[0], index=frame.index.levels[1][1:-1])
    if t_range:
        head = head.loc[t_range]
        filament = filament.loc[t_range]
    km = filament.k.mean(level=0)  
    ks = filament.k.unstack().dropna(axis=1).values
    ks_s = second_harm_tools.smooth_eigen(ks)

    s0 = int(round((len(ks.T)-1)*rel_s))
    head["kappa_mean"] = km
    head["kappa"] = ks.T[s0] 
    head["kappa_s"] = ks_s.T[s0]

    return head

def get_sea_urchin(path, rel_s=2/3, t_range=None):
    with h5py.File(path) as store:
        return _get_sea_urchin(store, rel_s, t_range)

def get_human(path, rel_s=2/3, t_range=None):

    ## Load Data
    data = load_xyz(open(path + "/trajectory.xyz"))
    df = convert_xyz_df_coord_index_to_column(data)
    df[df.X ==0] = np.nan
    df.index.names = ["time", "s"]
    tmp = df.reset_index()
    tmp.time /= 500
    df = tmp.set_index(["time", "s"])

    curv = -np.load(path + "/curvature.npy")[t_range]
    ks_s = second_harm_tools.smooth_eigen(curv)
    phi = np.unwrap(np.arctan2(df.xs(0,level=1).Y, df.xs(0,level=1).X))[t_range]
    head = pd.DataFrame({"phi": phi}, index=df.index.levels[0][t_range])
    head["kappa_mean"] = curv.mean(axis=1)

    s0 = int(round((len(curv.T)-1)*rel_s))

    head["kappa"] = curv.T[s0]
    head["kappa_s"] = ks_s.T[s0+1]

    return head
