# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib as mpl
from matplotlib import cm
from math import sqrt
from math import exp, expm1 , log
from numpy import random
import pandas as pd
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import random


np.random.seed()
random.seed()

def f_1 (x):
	return x**4 -x**3 -20*x**2 + x +1

def recuit_f1 (x0, t0, k, kp ,  tmax , A_rate_max ):	# t => time and T=> Temperature
	x=[x0]
	f=[f_1(x0)]
	x_all = [x0]
	f_all = [f_1(x0)]
	t=t0
	T = 1/t0
	A_rate = 1	# suppose qu'on commence par réduire la fonction de cout
	
	while t < tmax   : # rmq nb max iter implique une condition sur la fct de cout
		#or A_rate > A_rate_max
		xx = x[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1  )	
		ff = f_1(xx)
		if ff < f[-1]:
			x.append(xx)
			f.append(ff)
		else :
			if random.random() < kp * exp( -1 / (1000*T)):
				x.append(xx)
				f.append(ff)

		# Conservation de toutes les solutions		
		x_all.append(xx)
		f_all.append(ff)
						
		t+=1 	# Pas de convergence	
		T  = 1 / t 		
		A_rate = len(x)/t # nombre de mouvement réellement effectué par rapport au nombre d'iéttération total

	return  x,f,x_all,f_all		

###############################################################################################
#								MAIN														  #	
###############################################################################################
Which_question  = int(input('Which question ?	'))
if Which_question==1:
	X = np.arange(-6,6,0.1)
	fig = plt.figure(1)
	plt.plot(X, f_1(X))
	plt.xlabel('x')
	plt.ylabel('f(x)')	
	plt.show()
	
if Which_question==2:
	S = recuit_f1 ( -5, 1, 10, 0.5 ,  100 ,  10**-5 )
	X = np.arange(-6,6,0.1)
	fig = plt.figure(1)
	plt.plot(X, f_1(X))
	plt.plot(S[0], S[1] , c='red')
	plt.xlabel('x')
	plt.ylabel('f(x)')	
	plt.show()
