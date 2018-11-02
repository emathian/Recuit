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
	print('Trajet', Trajet)
	D = 0
	while len(Trajet) > 1 :
		#print('carte[Trajet[0]][0]', carte[Trajet[0]][0] ,  'carte[Trajet[1]][0] ', carte[Trajet[1]][0],  'carte[Trajet[0]][1]', 'carte[Trajet[1]][1]',carte[Trajet[1]][1])
		D += sqrt((carte[Trajet[0]][0]-carte[Trajet[1]][0])**2+(carte[Trajet[0]][1]-carte[Trajet[1]][1])**2 )
		Trajet.pop(0)
	return D	
def random_journey (journey):
	
	i =  np.random.randint(len(journey))
	j = np.random.randint(len(journey))
	old_journay_i = journey[i]
	journey[i]=journey[j]
	journey[j] = old_journay_i
	return journey


#macarte = (cities(3,3))
#print(macarte)
T1= [0,2,3,8,1,10]
#print(distance(T1, macarte))
print(random_journey(T1))

