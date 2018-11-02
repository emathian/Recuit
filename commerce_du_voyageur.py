# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from math import exp, expm1 , log
from numpy import random
#import pandas as pd
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import random
from scipy.stats import chisquare
from scipy.stats import ttest_ind
from scipy.stats import t
import statistics as s


np.random.seed()
random.seed()

def cities(L, N):
	return np.random.rand(N,2) * L

def distance (Trajet, carte):
	D = 0
	while len(Trajet) > 1 :
		D += sqrt((carte[Trajet[0]][0]-carte[Trajet[1]][0])**2+(carte[Trajet[0]][1]-carte[Trajet[1]][1])**2 )
		Trajet.pop(0)
	return D	
macarte = (cities(3,3))
print(macarte)
T1= [0,2,1]
print(distance(T1, macarte))

