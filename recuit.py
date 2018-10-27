# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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

def g (x,y):
	return x**4 -x**3 -20*x**2 + x +1 + y**4 -y**3 -20*y**2 + y +1


def recuit_f1 (x0, t0, k, kp ,  tmax , A_rate_max ):	# t => time and T=> Temperature
	x=[x0]
	f=[f_1(x0)]
	t=t0
	T = 1/t0
	A_rate = 1	# suppose qu'on commence par réduire la fonction de cout
	
	while t < tmax  and A_rate > A_rate_max : # rmq nb max iter implique une condition sur la fct de cout
		
		xx = x[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1  )	
		ff = f_1(xx)
		if ff < f[-1]:
			x.append(xx)
			f.append(ff)
		else :
			if random.random() < kp * exp( -1 / (1000*T)):
				x.append(xx)
				f.append(ff)
				
		t+=1 	# Pas de convergence	
		T  = 1 / t 		
		A_rate = len(x)/t # nombre de mouvement réellement effectué par rapport au nombre d'iéttération total
	
	return  x,f,t	
	


def recuit_g (x0, y0, t0, k, kp ,  tmax , A_rate_max ):	# t => time and T=> Temperature
	x=[x0]
	y=[y0]
	f=[g(x0,y0)]
	t=t0
	T = 1/t0
	A_rate = 1	# suppose qu'on commence par réduire la fonction de cout
	
	while t < tmax  and A_rate > A_rate_max : # rmq nb max iter implique une condition sur la fct de cout
		
		xx = x[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1  )	
		yy = y[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1  )	
		ff = g(xx,yy)
		if ff < f[-1]:
			x.append(xx)
			y.append(yy)
			f.append(ff)
		else :
			if random.random() < kp * exp( -1 / (1000*T)):
				x.append(xx)
				y.append(yy)
				f.append(ff)
				
		t+=1 	# Pas de convergence	
		T  = 1 / t 		
		A_rate = len(x)/t # nombre de mouvement réellement effectué par rapport au nombre d'iéttération total
	
	return  x,y,f,t			

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
	print('min de f1  :  ',min(f_1(X)))
	
if Which_question==2:
	S = recuit_f1 ( -5, 1, 10, 0.5 , 10000, 0.0001 )
	X = np.arange(-6,6,0.1)
	fig = plt.figure(1)
	plt.plot(S[0], S[1] , 'k-x')
	plt.plot(X, f_1(X),  c='red')
	plt.xlabel('x')
	plt.ylabel('f(x)')	


	#T = np.arange(0,1,0.01)
	#fig = plt.figure(2)
	#plt.plot(T, 0.1*np.exp(-1/(1000*T)),  c='g') # Aug variance
	#plt.plot(T, 10*np.exp(-1/(1000*T)),  c='b') # Dim variance
	#plt.xlabel('x')
	#plt.ylabel('f(x)')	
	#plt.show()

	#print(min(S[1]))
	#print((S[2]))

	S1 = recuit_f1 ( 0.5, 1, 15, 0.5 , 10000, 0.0001 )
	strS1 = ('x0 = 0.5 ; t0 = 1 ; k = 15 ; kp = 0.5 ; tmax =  10000' )
	S2 = recuit_f1 ( 0.5, 1, 1 , 0.5 , 10000, 0.0001 )
	strS2 = ('x0 = 0.5 ; t0 = 1 ; k = 1 ; kp = 0.5 ; tmax =  10000' )
	S3 = recuit_f1 ( 0.5, 1, 10, 0.8 , 10000, 0.0001 )
	strS3 = ('x0 = 0.5 ; t0 = 1 ; k = 10 ; kp = 0.8 ; tmax =  10000' )
	S4 = recuit_f1 ( 0.5, 1, 1 , 0.1 , 10000, 0.0001 )
	strS4 = ('x0 = 0.5 ; t0 = 1 ; k = 10 ; kp = 0.1 ; tmax =  10000' )
	L_sol=[]
	L_title=[]
	L_sol.append(S1)
	L_sol.append(S2)
	L_sol.append(S3)
	L_sol.append(S4)
	L_title.append(strS1)
	L_title.append(strS2)
	L_title.append(strS3)
	L_title.append(strS4)
	for i in range(0,5) :
		fig = plt.figure(i+3)
		plt.plot(L_sol[i][0], L_sol[i][1] , 'k-x')
		plt.plot(X, f_1(X),  c='red')	
		plt.xlabel('x')
		plt.title(L_title[i])
		plt.ylabel('f(x)')	
	plt.show()






if Which_question==3:
	S = recuit_f1 ( -5, 1, 10, 0.1 , 50, 0.001 )
	print(S[2])
	X = np.arange(-6,6,0.1)
	fig = plt.figure(1)
	plt.plot(S[0], S[1] , 'k-x')
	plt.plot(X, f_1(X),  c='red')
	plt.xlabel('x')
	plt.ylabel('f(x)')	
	plt.show()

if Which_question==4:
	X = np.arange(-5,5,0.2)
	Y = np.arange(-5,5,0.2)
	X, Y = np.meshgrid(X, Y)
	Z= g(X,Y)


	S = recuit_g (0, 0, 1, 1, 0.001 ,  10000 , 0.0001 )
	fig = plt.figure(0) #opens a figure environment
	ax = fig.gca(projection='3d') #to perform a 3D plot
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	ax.plot(S[0],S[1],S[2], c='red')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	

	
	S1 = recuit_g ( 0,0, 1, 1, 0.5 , 100, 0.0001 )
	strS1 = ('x0 = 0 ; y0 = 0 ;t0 = 1 ; k = 1 ; kp = 0.5 ; tmax =  100' )
	S2 = recuit_g ( 0,0, 1, 0.1 , 0.5 , 100, 0.0001 )
	strS2 = ('x0 = 0 ; y0 = 0 ; t0 = 1 ; k = 0.1 ; kp = 0.5 ; tmax =  100' )
	S3 = recuit_g ( 0,0, 1, 10, 0.2 , 100, 0.0001 )
	strS3 = ('x0 = 0 ; y0 = 0 ; t0 = 1 ; k = 10 ; kp = 0.2 ; tmax =  100' )
	S4 = recuit_g ( 0,0, 1, 1 , 0.1 , 100, 0.0001 )
	strS4 = ('x0 = 0 ; y0 = 0 ; t0 = 1 ; k = 10 ; kp = 0.1 ; tmax =  100' )
	L_sol=[]
	L_title=[]
	L_sol.append(S1)
	L_sol.append(S2)
	L_sol.append(S3)
	L_sol.append(S4)
	L_title.append(strS1)
	L_title.append(strS2)
	L_title.append(strS3)
	L_title.append(strS4)
	for i in range(0,4) :
		fig = plt.figure(i+1)
		ax = fig.gca(projection='3d') #to perform a 3D plot
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
		ax.plot(L_sol[i][0],L_sol[i][1],L_sol[i][2], c='red')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('g(x,y)')
		plt.title(L_title[i])
	plt.show()
	
