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
	journeyc =journey
	i =  np.random.randint(len(journey))
	j = np.random.randint(len(journey))
	old_journay_i = journey[i]
	journeyc[i]=journeyc[j]
	journeyc[j] = old_journay_i
	return journey

def recuit_traveller (country , journey0, t0,k, kp ,tmax, A_rate_max, m, cooling ):
	j = [journey0]
	
	d = [distance(journey0, country)]


	t = t0


	#tt_accepted = t0
	#t_accepted=[tt_accepted]
	Temp = [1/t0]
	T =Temp[-1]
	A_rate = 1	# suppose qu'on commence par réduire la fonction de cout
	while t < tmax  : # rmq nb max iter implique une condition sur la fct de cout #and A_rate > A_rate_max 
		## palier
		S = 0
		for i in range(m):
			jc= random_journey(np.copy(j[-1]))
			S+= distance(jc,country)- distance(j[-1],country)
		

		DE = 1/m * S
		jj = random_journey(np.copy(j[-1]))
		#print(jj)
		dd = distance(jj,country)
		if dd < d[-1]:
			j.append(jj)
			d.append(dd)
			TR  = 1 / t 	
			#tt_accepted += 1
			#t_accepted.append(tt_accepted)
			
		else :
			if random.random() < kp * exp( -DE / (1000*T)):
				j.append(jj)
				d.append(dd)
				#tt_accepted += 1
				#t_accepted.append(tt_accepted)


		t+=1 	# Pas de convergence

		if cooling == 'inv':
			T  = 1 / t 	
			Temp.append(T)
		elif cooling == 'inv_cube':
			T = 1/ t**3
			Temp.append(T)
		elif cooling == 'inv_log' :
			T = 1/log(t)	
			Temp.append(T)

		
		#A_rate = len(jj)/t # nombre de mouvement réellement effectué par rapport au nombre d'iéttération total
		#t_accepted.append(10000)
		#d.append(d[-1])
	return  j,d,Temp



def recuit_traveller_display (country , journey0, t0, k, kp ,tmax, m , cooling):
	# np.copy 
	j = [journey0]
	d = [distance(journey0, country)]
	t=t0
	Time= [t0]
	T = 1/t0
	Var_proba =[]	
	Var_perimetre =[]	

	sort_country = np.zeros((np.shape(country)[0], np.shape(country)[1]))
	for i,k in zip(j[-1], range(len(country))):
		sort_country[k,] = country[i,]
				
	fig = plt.figure()
	plt.plot(sort_country[:,0],sort_country[:,1],'-o')
	strS = 't = (%d) ; d =(%f)'  % (t, d[-1] )
	plt.title( strS )
	Title= 'voyage %d' % 0
	fig.savefig(Title) 
	
	A_rate = 1	# suppose qu'on commence par réduire la fonction de cout
	while t <= tmax   : # rmq nb max iter implique une condition sur la fct de cout
		## palier
		S = 0
		for i in range(m):

			jc= random_journey(np.copy(j[-1]))
			S+= distance(jc,country) - distance(j[-1],country)
			#print('i', i, 'jc',jc ,'j[-1]',j[-1],'S', S)
		#print('S loop', S)

		DE = 1/m * S
		jj = random_journey(np.copy(j[-1]))
		dd = distance(jj,country)
		if dd < d[-1]:
			j.append(jj)
			d.append(dd)
			Time.append(1/t)

		else :
			var_proba = kp * exp( -DE / (1000*T))
			Var_proba.append(var_proba)
			if random.uniform(0,1) < var_proba:
				j.append(jj)
				d.append(dd)
				Time.append(t)

			
		Time.append(tmax)
		distance_graph =d
		distance_graph.append(d[-1])
		sort_country = np.zeros((np.shape(country)[0], np.shape(country)[1]))
		if t% 200 == 0 : 
			for i,k in zip(j[-1], range(len(country))):
				sort_country[k,] = country[i,]
				
			fig = plt.figure()
			plt.plot(sort_country[:,0],sort_country[:,1],'-o')
			strS = 't = (%d) ; d =(%f)'  % (t, d[-1] )
			plt.title( strS )

			plt.pause(0.3)

			if t%400 ==0  :
				Title= 'voyage %d' % t
				fig.savefig(Title) 
		t+=1 	# Pas de convergence	
		if cooling == 'inv':
			T  = 1 / t 	
		elif cooling == 'inv_cube':
			T = 1/ t**3
		elif cooling == 'inv_log' :
			T = 1/log(t)			
	plt.ioff()   	
	plt.draw() 
	print('len d',len(d))
	Ind  = np.arange(len(d))
	Ind_proba = np.arange(len(Var_proba))

	fig = plt.figure(t + 1)
	plt.plot(Time, distance_graph )
	
	plt.ylabel('distance')	
	plt.xlabel('Nb iterations')	

	fig = plt.figure(t + 2)
	plt.plot(Ind , d)
	plt.ylabel('distance')	
	plt.xlabel('Nb iterations')	
	
	Title= 'distance %d' % 2
	fig.savefig(Title) 

	fig = plt.figure(t + 3)
	plt.plot(Ind_proba , Var_proba, c='navy')
	plt.ylabel('proba d accepeter une mauvaise solution')
	plt.ylim((-0.5,10))	
	plt.xlabel('Nb itteration')	
	
	Title= 'Var_PROBA %d' % 3
	fig.savefig(Title) 
		
	return  j,d,t		


	
	
Which_question  = int(input('Which question ?	'))
if Which_question==1 :
	macarte = (cities(3,10))
	print(macarte)
	T1= np.arange(0,10)
	#print(distance(T1, macarte))
	#display(T1, macarte)

if Which_question==2 :
	macarte = (cities(3,10))
	T1= np.arange(0,10)
	recuit_traveller_display(macarte, T1, 1,10, 0.5 ,10000, 5, 'inv' )

if Which_question==3 :
	macarte = (cities(3,10))
	print(macarte)
	T1= np.arange(0,10)
	s_inv = recuit_traveller(macarte, T1, 1,10, 0.5 ,10000,0.000001,  5 , 'inv')
	#print(s_inv[0])
	s_inv_cube = recuit_traveller(macarte, T1, 1,10, 0.5 ,10000,0.000001,   5 , 'inv_cube')
	s_log = recuit_traveller(macarte, T1, 1,10, 0.5 ,10000,0.000001,   5 , 'inv_log')


	sort_country_inv = np.zeros((np.shape(macarte)[0], np.shape(macarte)[1]))
	for i,k in zip(s_inv[0][-1], range(len(macarte))):
		sort_country_inv[k,] = macarte[i,]


	sort_country_inv_cube = np.zeros((np.shape(macarte)[0], np.shape(macarte)[1]))
	for i,k in zip(s_inv_cube[0][-1], range(len(macarte))):
		sort_country_inv_cube[k,] = macarte[i,]


	sort_country_log = np.zeros((np.shape(macarte)[0], np.shape(macarte)[1]))
	for i,k in zip(s_log[0][-1], range(len(macarte))):
		sort_country_log[k,] = macarte[i,]

	s_inv_min = min(s_inv[1])
	s_inv_cube_min = min(s_inv_cube[1])
	s_log_min = min(s_log[1])



	plt.figure()
	plt.subplot(331)
	plt.plot(sort_country_inv[:,0],sort_country_inv[:,1],'-', c='navy')
	strS = 'T = (%f) ; d =(%f)'  % (s_inv[2][-1], s_inv[1][-1])
	plt.title( strS )

	plt.subplot(332)
	plt.plot(sort_country_inv_cube[:,0],sort_country_inv_cube[:,1],'-', c='royalblue')
	strS = 'T = (%f) ; d =(%f)'  % (s_inv_cube[2][-1], s_inv_cube[1][-1])
	plt.title( strS )

	plt.subplot(333)
	plt.plot(sort_country_log[:,0],sort_country_log[:,1],'-', c= 'deepskyblue')
	strS = 'T = (%f) ; d =(%f)'  % (s_log[2][-1], s_log[1][-1])
	plt.title( strS )


	Ind_inv  = np.arange(len(s_inv[0]))
	Ind_inv_cube  = np.arange(len(s_inv_cube[0]))
	Ind_log = np.arange(len(s_log[0]))

	plt.subplot(334)
	plt.plot(Ind_inv,s_inv[1],'-',c='navy')
	plt.axhline(y=s_inv_min, color='r', linestyle='-')

	plt.subplot(335)
	plt.plot(Ind_inv_cube,s_inv_cube[1],'-',c='royalblue')
	plt.axhline(y=s_inv_cube_min, color='r', linestyle='-')
	#plt.xlim((0,10000))

	plt.subplot(336)
	plt.plot(Ind_log,s_log[1],'-', c= 'deepskyblue')
	plt.axhline(y=s_log_min, color='r', linestyle='-')

	X = np.arange(1,10001)
	T_inv = 1/X
	T_inv_cube =1 / X**3
	T_log = 1 / np.log(X)

	plt.subplot(337)
	plt.plot(X,np.log(T_inv),'-',c='navy')
	plt.xlabel("Nb iteration")
	plt.ylabel("log(1/t)")

	plt.subplot(338)	
	plt.plot(X,np.log(T_inv_cube),'-',c='royalblue')
	plt.xlabel("Nb iteration")
	plt.ylabel("log(1/t^3)")

	plt.subplot(339)
	plt.plot(X,np.log(T_log),'-', c= 'deepskyblue')
	plt.xlabel("Nb iteration")
	plt.ylabel("log(1/log(t))")

	plt.show()

if Which_question==4 :
	Dist_inv = []
	Dist_inv_cube = []
	Dist_log = []

	temps_inv = []
	temps_inv_cube = []
	temps_log = []
	for i in range(21):
		macarte = (cities(3,10))
		T1= np.arange(0,10)
		T1 = random_journey(T1)
		s_inv = recuit_traveller(macarte, T1, 1,10, 0.5 ,10000,0.000001,  5 , 'inv')
		s_inv_cube = recuit_traveller(macarte, T1, 1,10, 0.5 ,10000,0.000001,   5 , 'inv_cube')
		s_log = recuit_traveller(macarte, T1, 1,10, 0.5 ,10000,0.000001,   5 , 'inv_log')
		Dist_inv.append(s_inv[1][-1])
		Dist_inv_cube.append(s_inv_cube[1][-1])
		Dist_log.append(s_log[1][-1])
		temps_inv.append(len(s_inv[1]))
		temps_inv_cube.append(len(s_inv_cube[1]))
		temps_log.append(len(s_log[1]))

	print('Mean optimal distance for 1/t',  np.mean(Dist_inv))	
	print('Mean optimal distance for 1/t^3',  np.mean(Dist_inv_cube))		
	print('Mean optimal distance for 1/log(t)',  np.mean(Dist_log))
	# Merge_dist  = Dist_inv+ Dist_inv_cube + Dist_log
	# cooling = np.array(["1/t", "1/t^3", "1/log(t)"])
	# cooling  = np.repeat(Merge_dist, [len(Dist_inv), len(Dist_inv_cube), len(Dist_log)], axis=0)
	# data_dist = np.column_stack((cooling, Merge_dist))
	# boxplot(data_dist)	
	print('Dist inv', Dist_inv)
	print('Dist inv cube', Dist_inv_cube)
	print('Dist inv log', Dist_log)

	print('Temps inv', temps_inv)
	print('TEmps inv cube', temps_inv_cube)
	print('Temp inv log',temps_log)







