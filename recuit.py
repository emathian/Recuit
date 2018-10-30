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
from matplotlib import cm
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


def recuit_f1 (xd ,xp, t0, k, kp , tmax , A_rate_max ):	# t => time and T=> Temperature
	
	x0 =random.uniform( xd, xp )
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

def recuit_f1_p (x0, t0, k, kp, tmax , A_rate_max, m ):	# t => time and T=> Temperature
	x=[x0]
	f=[f_1(x0)]
	t=t0
	T = 1/t0
	A_rate = 1	# suppose qu'on commence par réduire la fonction de cout
	
	while t < tmax  and A_rate > A_rate_max : # rmq nb max iter implique une condition sur la fct de cout
		## palier
		S = 0
		for i in range(m):
			xc= x[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1)
			S+= f_1(xc) - f_1(x[-1])
		

		DE = 1/m * S
		xx = x[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1  )	
		ff = f_1(xx)
		if ff < f[-1]:
			x.append(xx)
			f.append(ff)
		else :
			if random.random() < kp * exp( -DE / (1000*T)):
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

def recuit_g_p (x0, y0,t0, k, kp, tmax , A_rate_max, m ):	# t => time and T=> Temperature
	x=[x0]
	y=[y0]
	f=[g(x0,y0)]
	t=t0
	T = 1/t0
	A_rate = 1	# suppose qu'on commence par réduire la fonction de cout
	
	while t < tmax  and A_rate > A_rate_max : # rmq nb max iter implique une condition sur la fct de cout
		## palier
		S = 0
		for i in range(m):
			xc= x[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1)
			yc= y[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1)
			S+= g(xc, yc) - g(x[-1], y[-1])
			
		DE = 1/m * S
		xx = x[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1  )	
		yy = y[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1  )	
		ff = g(xx,yy)
		if ff < f[-1]:
			x.append(xx)
			y.append(yy)
			f.append(ff)
		else :
			if random.random() < kp * exp( -DE / (1000*T)):
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
	S = recuit_f1 ( -6, 6, 1,10, 0.5 , 10000, 0.0001 )
	X = np.arange(-6,6,0.1)
	fig = plt.figure(1)
	plt.plot(S[0], S[1] , '-x', c= 'lightgrey')
	plt.plot(S[0][-1], S[1][-1] , 'x', c= 'c')
	plt.plot(X, f_1(X))
	plt.xlabel('x')
	plt.ylabel('f(x)')	
	
	print(S[0][-1], S[1][-1])

	F = min(f_1(X))

	k = np.arange(0,30,0.5)
	L_sol_ecart =[]
	for i in range(len(k)):
		S = recuit_f1 ( -6, 6, 1,k[i], 0.5 , 10000, 0.0001 )
		E = (abs(S[1][-1] -F))
		L_sol_ecart.append(E)

	fig = plt.figure(2)
	plt.plot(k ,L_sol_ecart,  'x')
	plt.xlabel('k')
	plt.ylabel('| f(x_min_calcule) - f(x_opt)|')	
	
	print(S[0][-1], S[1][-1])


	T = np.arange(1*10**-6 ,1/1000,0.00001)
	k = [1,5,10,15]
	fig = plt.figure(3)
	for i in k:
		f= i * np.exp(-1/(1000*T))
		plt.plot(T, f, c=cm.hot(i/15), label=i)
		plt.legend()
		plt.xlabel('T')
		plt.ylabel(' k.e^{-1/(1000*T)}')	
	

	kp = np.arange(0,1,0.01)
	L_sol_ecart =[]
	for i in range(len(kp)):
		S = recuit_f1 ( -6, 6, 1,10, kp[i] , 10000, 0.0001 )
		E = (abs(S[1][-1] -F))
		L_sol_ecart.append(E)

	fig = plt.figure(4)
	plt.plot(kp ,L_sol_ecart,  'x')
	plt.xlabel('kP')
	plt.ylabel('| f(x_min_calcule) - f(x_opt)|')	

	Tmax = np.arange(0,30000,100)
	L_sol_ecart =[]
	for i in range(len(Tmax)):
		S = recuit_f1 ( -6, 6, 1,10, 0.5 , Tmax[i], 10**-6 )
		E = (abs(S[1][-1] -F))
		L_sol_ecart.append(E)

	fig = plt.figure(5)
	plt.plot(Tmax ,L_sol_ecart,  'x')
	plt.xlabel('tmax')
	plt.ylabel('| f(x_min_calcule) - f(x_opt)|')
	plt.ylim((0,100))	
	plt.show()
	print(S[0][-1], S[1][-1])


if Which_question==3:
	S = recuit_f1 ( -5,5, 1, 12, 0.1 , 20000, 0.0001 )

	X = np.arange(-6,6,0.1)
	fig = plt.figure(1)
	plt.plot(S[0][1], S[1][1] , '-o' ,c='green')
	plt.plot(S[0], S[1] , '-x' ,c='lightgrey')
	plt.plot(S[0][-1], S[1][-1] , '-o' ,c='red')
	plt.plot(S[0][1], S[1][1] , '-o' ,c='green')
	plt.plot(X, f_1(X))

	plt.xlabel('x')
	plt.ylabel('f(x)')	
	plt.show()

if Which_question==4:
	X = np.arange(-5,5,0.2)
	Y = np.arange(-5,5,0.2)
	X, Y = np.meshgrid(X, Y)
	Z= g(X,Y)
	print('min g :',Z)


	S = recuit_g (0, 0, 1, 1, 0.001 ,  10000 , 0.0001 )
	fig = plt.figure(0) #opens a figure environment
	ax = fig.gca(projection='3d') #to perform a 3D plot
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	ax.plot(S[0],S[1],S[2], c='red')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	

	
	S1 = recuit_g ( 0,0, 1, 1, 0.5 , 10000, 0.0001 )
	strS1 = ('x0 = 0 ; y0 = 0 ;t0 = 1 ; k = 1 ; kp = 0.5 ; tmax =  100' )
	S2 = recuit_g ( 0,0, 1, 0.1 , 0.5 , 10000, 0.0001 )
	strS2 = ('x0 = 0 ; y0 = 0 ; t0 = 1 ; k = 0.1 ; kp = 0.5 ; tmax =  100' )
	S3 = recuit_g ( 0,0, 1, 10, 0.2 , 10000, 0.0001 )
	strS3 = ('x0 = 0 ; y0 = 0 ; t0 = 1 ; k = 10 ; kp = 0.2 ; tmax =  100' )
	S4 = recuit_g ( 0,0, 1, 1 , 0.1 , 10000, 0.0001 )
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
		ax.plot(L_sol[i][0][-1],L_sol[i][1][-1],L_sol[i][2][-1], '-o',c='red')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('g(x,y)')
		plt.title(L_title[i])
	plt.show()
	

if Which_question==5:

	S = recuit_f1_p ( -5, 1, 10, 0.5 , 10000, 0.0001 , 5)
	S1 = recuit_f1( -5, 1, 10, 0.5 , 10000, 0.0001 )
	X = np.arange(-6,6,0.1)
	fig = plt.figure(0)
	plt.plot(S[0][-1], S[1][-1] , 'k-x')
	plt.plot(S1[0][-1], S1[1][-1] , 'c-x')
	plt.plot(X, f_1(X),  c='red')
	plt.xlabel('x')
	plt.ylabel('f(x)')	
	
	print(S[1][-1])
	print(S1[1][-1])

	k=[15,1,10,10]
	kp=[0.5,0.5,0.8,0.1]
	for i in range(len(k)):
		S= recuit_f1 ( 0.5, 1,k[i], kp[i] , 10000, 0.0001 )
		S1= recuit_f1_p ( 0.5, 1,k[i], kp[i] , 10000, 0.0001 ,5)
		strS = 'x0 = 0.5 ; t0 = 1 ; k = (%f); kp = (%f) ; tmax =  10000'  % (k[i] ,kp[i] )
		print('len S', len(S[0]), 'lenS1', len(S1[0]))
		fig = plt.figure(i+1)
		plt.plot(S[0][-1], S[1][-1] , 'k-x')
		plt.plot(S1[0][-1], S1[1][-1] , 'c-x')
		plt.plot(X, f_1(X),  c='red')	
		plt.xlabel('x')
		plt.title(strS)
		plt.ylabel('f(x)')	
	
	plt.show()


if Which_question==6:
	X = np.arange(-5,5,0.2)
	Y = np.arange(-5,5,0.2)
	X, Y = np.meshgrid(X, Y)
	Z= g(X,Y)


	S = recuit_g_p ( 0,0, 1, 10, 0.5 , 10000, 0.0001 , 5)
	S1 = recuit_g( 0,0, 1, 10, 0.5 , 10000, 0.0001 )

	fig = plt.figure(0)
	ax = fig.gca(projection='3d') #to perform a 3D plot
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	ax.plot(S[0][-1],S[1][-1],S[2][-1], '-o', c='red')
	ax.plot(S1[0][-1],S1[1][-1],S1[2][-1], '-o',c='cyan')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	
	
	print(S[1][-1])
	print(S1[1][-1])

	plt.show()