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

def recuit_traveller (country , journey0, k, kp ,tmax, A_rate_max, m ):
	j = [journey0]
	d = [distance(journey0, country)]
	t=t0
	T = 1/t0
	A_rate = 1	# suppose qu'on commence par réduire la fonction de cout
	while t < tmax  and A_rate > A_rate_max : # rmq nb max iter implique une condition sur la fct de cout
		## palier
		S = 0
		for i in range(m):
			jc= random_journey(j[-1])
			S+= distance(jc) - distance(j[-1])
		

		DE = 1/m * S
		jj = random_journey(j[-1])
		dd = distance(jj)
		if dd < d[-1]:
			j.append(jj)
			d.append(dd)
		else :
			if random.random() < kp * exp( -DE / (1000*T)):
				j.append(jj)
				d.append(dd)
				
		t+=1 	# Pas de convergence	
		T  = 1 / t 		
		A_rate = len(x)/t # nombre de mouvement réellement effectué par rapport au nombre d'iéttération total
	
	return  j,d,t		

#macarte = (cities(3,3))
#print(macarte)
T1= [0,2,3,8,1,10]
#print(distance(T1, macarte))
print(random_journey(T1))

