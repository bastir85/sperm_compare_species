# -*- coding: utf-8 -*-
"""
Modified: 29.6.15

@author: Guglielmo Saggiorato <g.saggiorato@fz-juelich.de> or <astyonax@gmail.com>
"""
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s - %(name)s.%(funcName)s  - %(levelname)s - %(message)s')
logger = logging.getLogger('root')
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)


def compute_curvature_xyz(xyz="",x="",y=""):
    """
    computes the signed curvature essentially by the cross product of
    neighbouring bonds
    input: xyz[#timestep,#particleid,#dimensions]
    example for 30 beads, in 3D and for 10 timesteps, xyz.shape==(10,30,3)
    output: curvature[#timestep,#particleid]
    """

    if xyz:
      dx=xyz[:,:,0]
      dy=xyz[:,:,1]
    else:
      dx=x
      dy=y

    ax=dx[:,:-2]-dx[:,1:-1]
    ay=dy[:,:-2]-dy[:,1:-1]
    n=(ax**2+ay**2)**.5

    bx=dx[:,2:]-dx[:,1:-1]
    by=dy[:,2:]-dy[:,1:-1]
    n=(bx**2+by**2)**.5

    cx=dx[:,2:]-dx[:,:-2]
    cy=dy[:,2:]-dy[:,:-2]

    az=bz=cz=0

    a=(ax**2+ay**2)**.5
    b=(bx**2+by**2)**.5
    c=(cx**2+cy**2)**.5

    A=(by*az-bz*ay)
    B=(bz*ax-bx*az)
    C=(bx*ay-by*ax)
    delta = 0.5 * (A+B+C)
    curv= -(4*delta/(a*b*c))

    return curv
