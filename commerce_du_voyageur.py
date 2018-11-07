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
		#print('carte[Trajet[0]][0]', carte[Trajet[0]][0] ,  'carte[Trajet[1]][0] ', carte[Trajet[1]][0],  'carte[Trajet[0]][1]', 'carte[Trajet[1]][1]',carte[Trajet[1]][1])
		D += sqrt((carte[Trajet[0]][0]-carte[Trajet[1]][0])**2+(carte[Trajet[0]][1]-carte[Trajet[1]][1])**2 )
		#Trajet.pop(0)
		Trajet = np.delete(Trajet, 0, 0)

	return D	
def random_journey (journey):
	# Faire une copie de journey
	i =  np.random.randint(len(journey))
	j = np.random.randint(len(journey))
	old_journay_i = journey[i]
	journey[i]=journey[j]
	journey[j] = old_journay_i
	return journey

def recuit_traveller (country , journey0, t0,k, kp ,tmax, A_rate_max, m ):
	j = [journey0]
	d = [distance(journey0, country)]
	t=t0
	T = 1/t0
	A_rate = 1	# suppose qu'on commence par réduire la fonction de cout
	while t < tmax  : # rmq nb max iter implique une condition sur la fct de cout #and A_rate > A_rate_max 
		## palier
		S = 0
		for i in range(m):
			jc= random_journey(j[-1])
			S+= distance(jc,country) - distance(j[-1],country)
		

		DE = 1/m * S
		jj = random_journey(j[-1])
		dd = distance(jj,country)
		if dd < d[-1]:
			j.append(jj)
			d.append(dd)
		else :
			if random.random() < kp * exp( -DE / (1000*T)):
				j.append(jj)
				d.append(dd)
		

		t+=1 	# Pas de convergence	
		T  = 1 / t 		
		#A_rate = len(jj)/t # nombre de mouvement réellement effectué par rapport au nombre d'iéttération total
		
	return  j,d,t		



def recuit_traveller_display (country , journey0, t0, k, kp ,tmax, m ):
	# np.copy 
	j = [journey0]
	d = [distance(journey0, country)]
	t=t0
	T = 1/t0
	A_rate = 1	# suppose qu'on commence par réduire la fonction de cout
	while t < tmax   : # rmq nb max iter implique une condition sur la fct de cout
		## palier
		S = 0
		for i in range(m):
			jc= random_journey(j[-1])
			S+= distance(jc,country) - distance(j[-1],country)
		

		DE = 1/m * S
		jj = random_journey(j[-1])
		dd = distance(jj,country)
		if dd < d[-1]:
			j.append(jj)
			d.append(dd)
		else :
			if random.uniform(0,1) < kp * exp( -DE / (1000*T)):
				j.append(jj)
				d.append(dd)

		sort_country = np.zeros((np.shape(country)[0], np.shape(country)[1]))
		if t% 200 == 0: 
			for i,k in zip(j[-1], range(len(country))):
				sort_country[k,] = country[i,]
				
			fig = plt.figure()
			plt.plot(sort_country[:,0],sort_country[:,1],'-o')
			strS = 't = (%f) ; d =(%d)'  % (t, d[-1] )
			plt.title( strS )
			plt.pause(0.3)

		t+=1 	# Pas de convergence	
		T  = 1 / t 		
	plt.ioff()   	
	plt.draw() 	
	return  j,d,t		


	
	


macarte = (cities(3,10))
print(macarte)
T1= [0,3,4,6,1,2,9,7,8,5]
#print(distance(T1, macarte))
#display(T1, macarte)




S =recuit_traveller_display (macarte, T1, 1,10, 0.5 ,10000, 5 )
print(S[0])
# s = recuit_traveller_display (macarte, T1, 1,10, 0.5 ,10000,  5 ) # a_rate plus d'influence 
# ind = np.arange(len(s[0]))
# sort_country = np.zeros((np.shape(macarte)[0], np.shape(macarte)[1]))
		
# for i,k in zip(s[0][-1], range(len(macarte))):
# 	sort_country[k,] = macarte[i,]

# fig = plt.figure(1)
# plt.plot(sort_country[:,0],sort_country[:,1],'-o')


# fig = plt.figure(2)
# plt.plot(ind,s[1])
# plt.show()


#fig = plt.figure(1)
#plt.plot(macarte[:,0],macarte[:,1],'-o')
# plt.xlabel('x')
# plt.ylabel('f(x)')	
#plt.show()	

print(random_journey(T1))
