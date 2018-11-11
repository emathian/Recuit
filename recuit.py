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
from scipy.stats import chisquare
from scipy.stats import ttest_ind
from scipy.stats import t
import statistics as s


np.random.seed()
random.seed()

def f_1 (x):
	return x**4 -x**3 -20*x**2 + x +1

def g (x,y):
	return x**4 -x**3 -20*x**2 + x +1 + y**4 -y**3 -20*y**2 + y +1


def recuit_f1 ( xd,xp,t0, k, kp , tmax , A_rate_max ):	# t => time and T=> Temperature
	#xd ,xp,
	#x=[x0]

	x0 =random.uniform( xd, xp )
	print(x0)
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
			if random.uniform(0,1) < kp * exp( -1 / (1000*T)):
				x.append(xx)
				f.append(ff)
				
		t+=1 	# Pas de convergence	
		T  = 1 / t 		
		A_rate = len(x)/t # nombre de mouvement réellement effectué par rapport au nombre d'iéttération total
	
	return  x,f,t	

def recuit_f1_p (xp, xd, t0, k, kp, tmax , A_rate_max, m ):	# t => time and T=> Temperature
	x0 =random.uniform( xd, xp )

	#
	x=[x0]
	#print('x0 ', x0)
	f=[f_1(x0)]
	t=t0
	T = 1/t0
	A_rate = 1	# suppose qu'on commence par réduire la fonction de cout
	
	while t < tmax  : # rmq nb max iter implique une condition sur la fct de cout #and A_rate > A_rate_max 
		## palier
		S = 0
		for i in range(m):
			xc= x[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1)
			#print('xc', xc, 'f_1(xc)', f_1(xc) ,'f_1(x[-1])', f_1(x[-1]) )
			S+= f_1(xc) - f_1(x[-1])
			#print('S', S )

		DE = 1/m * S
		#print('DE', DE)
		xx = x[-1] + np.random.normal(0, sqrt(k* exp(-1/(1000*T))) ,1  )	
		ff = f_1(xx)
		if ff < f[-1]:
			x.append(xx)
			f.append(ff)
		else :
			if random.uniform(0,1) < kp * exp( -DE / (1000*T)):
				x.append(xx)
				f.append(ff)

				
		t+=1 	# Pas de convergence	
		T  = 1 / t 		
		#A_rate = len(x)/t # nombre de mouvement réellement effectué par rapport au nombre d'iéttération total
	
	return  x,f,t		


def recuit_g (xd,xp, yd,yp, t0, k, kp ,  tmax , A_rate_max ):	# t => time and T=> Temperature
	x0 = random.uniform( xd, xp )
	y0= random.uniform( yd, yp )
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
			if random.uniform(0,1) < kp * exp( -1 / (1000*T)):
				x.append(xx)
				y.append(yy)
				f.append(ff)
				
		t+=1 	# Pas de convergence	
		T  = 1 / t 		
		A_rate = len(x)/t # nombre de mouvement réellement effectué par rapport au nombre d'iéttération total
	
	return  x,y,f,t		

def recuit_g_p (xd, xp , yd ,yp,t0, k, kp, tmax , A_rate_max, m ):	# t => time and T=> Temperature
	x0 = random.uniform( xd, xp )
	y0= random.uniform( yd, yp )

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
			if random.uniform(0,1) < kp * exp( -DE / (1000*T)):
				x.append(xx)
				y.append(yy)
				f.append(ff)
				
		t+=1 	# Pas de convergence	
		T  = 1 / t 		
		A_rate = len(x)/t # nombre de mouvement réellement effectué par rapport au nombre d'iéttération total
	
	return  x,y,f,t				

def stat(vp , F , n_rep , f , param , k_default, kp_default , tmax_default, sup_inf ):
	
	#n_rep = 15
	#kp = np.arange(0,1,0.02)
	lvp = len(vp)
	E =np.zeros((n_rep, lvp))
	Min_Inf =[]
	Min_sup =[]
	for i in range(n_rep):
		for j in range(lvp):
			if f=='f' and param=='k':
				#print('ok')
				S = recuit_f1 ( -5, 5, 1, vp[j] , kp_default , tmax_default, 0.0001 )
				E[i][j] = (abs(S[1][-1] -F))

			elif f=='f' and param=='kp':
				S = recuit_f1 ( -5, 5, 1, k_default , vp[j] , tmax_default, 0.0001 )
				E[i][j] = (abs(S[1][-1] -F))	
			elif f=='g' and param=='k':
				#print('ok')
				S = recuit_g ( -3, 3, -3,3,1, vp[j], kp_default , tmax_default, 0.0001 )
				E[i][j] = (abs(S[2][-1] -F))
				#print(E[i][j])
			elif f=='g' and param=='kp':
				S = recuit_g ( -3, 3, -3,3,1, k_default, vp[j] , tmax_default, 0.0001 )
				E[i][j] = (abs(S[2][-1] -F))		
			else :
				return ('unknown function')
		
	print(int(lvp/2))
	n_col_score = int(lvp/2)
	
	Score_inf = np.zeros((n_rep, n_col_score))
	for i in  range(n_rep):	
		for j in range(n_col_score):	
			if E[i][j] <2 and E[i][j]>-2 :
				Score_inf[i][j]=1

	Score_sup = np.zeros((n_rep, n_col_score))
	for i in  range(n_rep):	
		for j in range(n_col_score):	
			if E[i][j+n_col_score] <2 and E[i][j+n_col_score]>-2 :
				Score_sup[i][j]=1
	print('Score inf shape', np.shape(Score_inf))
	Sum_score_inf = np.sum(Score_inf,axis=0)
	Sum_score_sup = np.sum(Score_sup,axis=0)
	# ## T test unilateral 
	# ## H0  : S_k>20 > S_k<20
	mInf = s.mean(Sum_score_inf)
	mSup = s.mean(Sum_score_sup)
	vinf = s.pvariance(Sum_score_inf)
	vsup = s.pvariance(Sum_score_sup)
	if sup_inf == 0:
		# sup > inf
		T_test = (mSup - mInf )/ (sqrt(vinf/len(Sum_score_inf) + vsup/len(Sum_score_sup)))
	if sup_inf == 1:
		# inf > sup
		T_test = (mInf - mSup )/ (sqrt(vinf/len(Sum_score_inf) + vsup/len(Sum_score_sup)))


	pvB =2*( 1-(t.cdf(abs(T_test),len(Sum_score_inf) +len(Sum_score_sup) -2)) )
	pvU =  1-(t.cdf(T_test,len(Sum_score_inf) +len(Sum_score_sup) -2)) 

	print('Return Score inf' , Score_inf,
		'Return Score inf' , Score_sup,
		'Vecteur du nombre de succes sur n repetition partie inferieure : ', Sum_score_inf ,'\n',
		'Vecteur du nombre de succes sur n repetition partie superieure : ', Sum_score_sup ,'\n',
		'Moyenne de succes partie inferieure : ', mInf ,'\n',
		'Moyenne de succes partie superieur : ', mSup ,'\n',
		'Variance de succes partie inferieure : ', vinf ,'\n',
		'Variance de succes partie superieur : ', vsup ,'\n',
		'Statistique de student : ' , T_test,'\n',
		'Pvalue bilaterale :' , pvB , '\n',
		'Pvalue Unilateral  :', pvU )

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
	print('min de f1  :  ',(f_1(-2.823)))
if Which_question==2:
	S = recuit_f1 ( -6, 6, 1,10, 0.5 , 10000, 0.0001 )
	X = np.arange(-6,6,0.1)
	# fig = plt.figure(1)
	# plt.plot(S[0], S[1] , '-x', c= 'lightgrey')
	# plt.plot(S[0][-1], S[1][-1] , 'x', c= 'c')
	# plt.plot(X, f_1(X))
	# plt.xlabel('x')
	# plt.ylabel('f(x)')	
	
	print(S[0][-1], S[1][-1])

	F = min(f_1(X))

	# k = np.arange(0,100,0.5)
	# L_sol_ecart =[]
	# for i in range(len(k)):
	# 	S = recuit_f1 ( -6, 6, 1,k[i], 0.5 , 10000, 0.0001 )
	# 	E = (abs(S[1][-1] -F))
	# 	L_sol_ecart.append(E)

	# fig = plt.figure(2)
	# plt.plot(k ,L_sol_ecart,  'x')
	# plt.xlabel('k')
	# plt.ylabel('| f(x_min_calcule) - f(x_opt)|')	
	
	# print(S[0][-1], S[1][-1])


	# T = np.arange(1*10**-6 ,1/1000,0.00001)
	# k = [1,5,10,15]
	# fig = plt.figure(3)
	# for i in k:
	# 	f= i * np.exp(-1/(1000*T))
	# 	plt.plot(T, f, c=cm.hot(i/15), label=i)
	# 	plt.legend()
	# 	plt.xlabel('T')
	# 	plt.ylabel(' k.e^{-1/(1000*T)}')	
	

	# kp = np.arange(0,1,0.01)
	# L_sol_ecart =[]
	# for i in range(len(kp)):
	# 	S = recuit_f1 ( -6, 6, 1,10, kp[i] , 10000, 0.0001 )
	# 	E = (abs(S[1][-1] -F))
	# 	L_sol_ecart.append(E)

	# fig = plt.figure(4)
	# plt.plot(kp ,L_sol_ecart,  'x')
	# plt.xlabel('kP')
	# plt.ylabel('| f(x_min_calcule) - f(x_opt)|')	

	# Tmax = np.arange(0,30000,100)
	# L_sol_ecart =[]
	# for i in range(len(Tmax)):
	# 	S = recuit_f1 ( -6, 6, 1,10, 0.5 , Tmax[i], 10**-6 )
	# 	E = (abs(S[1][-1] -F))
	# 	L_sol_ecart.append(E)

	# fig = plt.figure(5)
	# plt.plot(Tmax ,L_sol_ecart,  'x')
	# plt.xlabel('tmax')
	# plt.ylabel('| f(x_min_calcule) - f(x_opt)|')
	# plt.ylim((0,100))	
	# plt.show()
	# print(S[0][-1], S[1][-1])
	

	#print('k = np.arange(0,15,0.5)')
	#print('F est le min global')
	k = np.arange(0,30,0.5)
	#kp = np.arange(0,1,0.02)

	print(stat( k, F  , 40,'g', 'k' , 10 , 0.5, 10000 ,0 ) )
	#print( stat(kp, F  ,40, 'f','kp' , 10 , 0.5, 10000 , 1))

if Which_question==3:
	S = recuit_f1 ( -5,5, 1, 0.1, 0 , 20000, 0.0001 )

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
	
	# print(S)
	Z= g(X,Y)
	M1 = g(3.548, 3.548)
	M2 = g(3.548, -2.823)
	M3 = g(-2.823,3.548)
	M4 = g(-2.823,-2.823)
	F= Z.min()
	# print(F)
	# print(M1 , M2 , M3 , M4)
	succes = 0
	for i in range (40):
		S = recuit_g ( -1, 1, -1,1 ,1,  15, 0.1 , 20000, 0.0001 )
		if abs(S[2][-1] - F) < 2 :
			succes += 1
	print('NOmbre de succes ' , succes)		
	#S = recuit_g (0, 0, 1, 1, 0.001 ,  10000 , 0.0001 )
	# fig = plt.figure() #opens a figure environment
	# ax = fig.gca(projection='3d') #to perform a 3D plot
	# ax.scatter(3.548,3.548,M1, '-o',c='red', s=200,label='(x,y,z)=(3.548,3.548, %d)' %M1)
	# ax.scatter(3.548,-2.823,M2,'-o' ,c='pink', s=200,label='(x,y,z)=(3.548,-2.823, %d)' %M2)
	# ax.scatter(-2.823,3.548,M3,'-o' ,c='orange', s=200,label='(x,y,z)=(-2.823,3.548, %d)' %M3)
	# ax.scatter(-2.823,-2.823,M4,'-o' ,c='yellow', s=200,label='(x,y,z)=(-2.823,-2.823, %d)' %M4)
	# ax.scatter(S[0][-1], S[1][-1],S[2][-1], '-o' ,c='green', s=200,label='(x,y,z)=(-2.823,-2.823, %d)' %M4)
	# surf = ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5) #plot definition and options
	
	# ax.plot(S[0],S[1],S[2] ,c='yellow')
	# plt.legend()
	# ax.set_xlabel('x')
	# ax.set_ylabel('y')
	# ax.set_zlabel('z')
	# plt.show()



	##k = np.arange(0,30,0.5) 
	##kp = np.arange(0.5,1,0.02)
	# print('kp = np.arange(0,1,0.02)')

	########## print(stat( k, F  , 14,'g', 'k' , 10 , 0.5, 10000 ,0 ) )
	

	##print( stat(k, F  ,40, 'f','k' , 10 , 0.5, 10000 , 1))



	######### k test ########################################
	# k = np.arange(0,30,0.5)
	# E =np.zeros((15, len(k)))
	# K = []
	# Min_Inf =[]
	# Min_sup =[]
	# for i in range(15):
	# 	for j in range(len(k)):
	# 		S = recuit_g ( -3, 3, -3,3,1,k[j], 0.5 , 10000, 0.0001 )
	# 		e = (abs(S[2][-1] -F))
	# 		K.append(k[j])
	# 		E[i][j] = e
	# print('len de k',len(k))	
	# print('len de E', E, 'E shape',np.shape(E))
	# print('len de K',len(K))

	# Score_inf = np.zeros((15, 30))
	# for i in  range(np.shape(E)[0]):	
	# 	for j in range(30):	
	# 		if E[i][j] <2 and E[i][j]>-2 :
	# 			Score_inf[i][j]=1

	# Score_sup = np.zeros((15, 30))
	# for i in  range(np.shape(E)[0]):	
	# 	for j in range(30):	
	# 		if E[i][j+30] <2 and E[i][j+30]>-2 :
	# 			Score_sup[i][j]=1

	# print(Score_inf)
	# print(Score_sup)	
	# Sum_score_inf = np.sum(Score_inf,axis=1)
	# Sum_score_sup = np.sum(Score_sup,axis=1)
	# # ## T test unilateral 
	# # ## H0  : S_k>20 > S_k<20
	# print('Sum_score_inf',Sum_score_inf)
	# print('Sum_score_sup',Sum_score_sup)
	# mInf = s.mean(Sum_score_inf)
	# print(mInf)
	# mSup = s.mean(Sum_score_sup)
	# print(mSup)
	# vinf = s.pvariance(Sum_score_inf)
	# vsup = s.pvariance(Sum_score_sup)
	# print(vinf)
	# print(vsup)
	# T_stat = (mSup - mInf )/ (sqrt(vinf/len(Sum_score_inf) + vsup/len(Sum_score_sup)))
	
	# pvU = 1-(t.cdf(T_stat,len(Sum_score_inf) +len(Sum_score_sup) -2))
	# print('pvalue : ', pvU)

	######### kp test ########################################




	# ######################################### Tmax

	# tmax = np.arange(0,20000,20)
	# for i in range(len(tmax)):
	# 	S = recuit_g ( -3, 3, -3,3,1,10, 0.5, tmax[i], 0.0001 )
	# 	E = (abs(S[2][-1] -F))
	# 	L_sol_ecart.append(E)
		

	# fig = plt.figure(3)
	
	# plt.xlabel('kp')
	# plt.axhline(y=0, xmin=0, xmax=30, c='red', label="Min  = -266")
	# plt.axhline(y=59, xmin=0, xmax=30, c='pink', label="Min  = -208")
	# plt.axhline(y=116, xmin=0, xmax=30, c='yellow', label="Min  = -150")
	# plt.plot(tmax ,L_sol_ecart,  'x')
	# plt.ylabel('| f(x_min_calcule) - f(x_opt)|')	




	# kp = np.arange(0,1.02,0.02)
	# print(kp)
	# Min_opt_kInf = 0
	# Min_opt_kSup = 0-
	# Min_Inf= []
	# Min_sup =[]
	# L_sol_ecart =[]

	# for i in range(len(kp)):
	# 	S = recuit_g ( -3, 3, -3,3,1,10, kp[i] , 10000, 0.0001 )
	# 	E = (abs(S[2][-1] -F))
	# 	L_sol_ecart.append(E)
		
	# fig = plt.figure(3)
	
	# plt.xlabel('kp')
	# plt.axhline(y=0, xmin=0, xmax=30, c='red', label="Min  = -266")
	# plt.axhline(y=59, xmin=0, xmax=30, c='pink', label="Min  = -208")
	# plt.axhline(y=116, xmin=0, xmax=30, c='yellow', label="Min  = -150")
	# plt.plot(kp ,L_sol_ecart,  'x')
	# plt.ylabel('| f(x_min_calcule) - f(x_opt)|')	


	# for j in range(15):
	# 	for i in range(len(kp)):
	# 		S = recuit_g ( -3, 3, -3,3,1,10, kp[i] , 10000, 0.0001 )
	# 		E = (abs(S[2][-1] -F))
	# 		L_sol_ecart.append(E)
	# 		if E <1 and E>-1 and kp[i] <0.5 :
	# 			Min_opt_kInf += 1
	# 		elif E <1 and E>-1 and kp[i] >0.5 :
	# 			Min_opt_kSup += 1
	# 	Min_Inf.append(Min_opt_kInf)		
	# 	Min_sup.append(Min_opt_kSup)	

	# # ## T test unilateral 
	# # ## H0  : S_k>20 > S_k<20
	# print(Min_Inf)
	# print(Min_sup)
	# mInf = s.mean(Min_Inf)
	# print(mInf)
	# mSup = s.mean(Min_sup)
	# print(mSup)
	# vinf = s.pvariance(Min_Inf)
	# print(vinf)
	# vsup = s.pvariance(Min_sup)
	# print(vsup)
	# T = (mSup - mInf) / (sqrt(vinf/len(Min_Inf)+ vsup/len(Min_sup)))
	# print(T)
	# pvU = 1-(t.cdf(T,len(Min_Inf)+ len(Min_sup)-2))
	# print('pvalue : ', pvU)

	
	# print(S[0][-1], S[1][-1])

	
	# S1 = recuit_g ( 0,0, 1, 1, 0.5 , 10000, 0.0001 )
	# strS1 = ('x0 = 0 ; y0 = 0 ;t0 = 1 ; k = 1 ; kp = 0.5 ; tmax =  100' )
	# S2 = recuit_g ( 0,0, 1, 0.1 , 0.5 , 10000, 0.0001 )
	# strS2 = ('x0 = 0 ; y0 = 0 ; t0 = 1 ; k = 0.1 ; kp = 0.5 ; tmax =  100' )
	# S3 = recuit_g ( 0,0, 1, 10, 0.2 , 10000, 0.0001 )
	# strS3 = ('x0 = 0 ; y0 = 0 ; t0 = 1 ; k = 10 ; kp = 0.2 ; tmax =  100' )
	# S4 = recuit_g ( 0,0, 1, 1 , 0.1 , 10000, 0.0001 )
	# strS4 = ('x0 = 0 ; y0 = 0 ; t0 = 1 ; k = 10 ; kp = 0.1 ; tmax =  100' )

	# L_sol=[]
	# L_title=[]
	# L_sol.append(S1)
	# L_sol.append(S2)
	# L_sol.append(S3)
	# L_sol.append(S4)
	# L_title.append(strS1)
	# L_title.append(strS2)
	# L_title.append(strS3)
	# L_title.append(strS4)
	# for i in range(0,4) :
	# 	fig = plt.figure(i+1)
	# 	ax = fig.gca(projection='3d') #to perform a 3D plot
	# 	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	# 	ax.plot(L_sol[i][0][-1],L_sol[i][1][-1],L_sol[i][2][-1], '-o',c='red')
	# 	ax.set_xlabel('x')
	# 	ax.set_ylabel('y')
	# 	ax.set_zlabel('g(x,y)')
	# 	plt.title(L_title[i])
	plt.show()
	

if Which_question==5:

	S = recuit_f1_p ( -5,5, 1, 10, 0.5 , 10000, 0.0001 , 5)
	#S1 = recuit_f1( -5,1, 10, 0.5 , 10000, 0.0001 )
	X = np.arange(-6,6,0.1)
	fig = plt.figure(0)
	plt.plot(S[0], S[1] , 'k-x')
	#plt.plot(S1[0][-1], S1[1][-1] , 'c-x')
	plt.plot(X, f_1(X),  c='red')
	plt.xlabel('x')
	plt.ylabel('f(x)')	
	
	print(S[1][-1])
	

	# k=[15,1,10,10]
	# kp=[0.5,0.5,0.8,0.1]
	# for i in range(len(k)):
	# 	#S= recuit_f1 ( 0.5, 1,k[i], kp[i] , 10000, 0.0001 )
	# 	S1= recuit_f1_p ( 0.5, 1,k[i], kp[i] , 10000, 0.0001 ,5)
	# 	strS = 'x0 = 0.5 ; t0 = 1 ; k = (%f); kp = (%f) ; tmax =  10000'  % (k[i] ,kp[i] )
	# 	print('len S', len(S[0]), 'lenS1', len(S1[0]))
	# 	fig = plt.figure(i+1)
	# 	#plt.plot(S[0], S[1], 'k-x')
	# 	plt.plot(S1[0], S1[1], 'c-x')
	# 	plt.plot(X, f_1(X),  c='red')	
	# 	plt.xlabel('x')
	# 	plt.title(strS)
	# 	plt.ylabel('f(x)')	
	
	# plt.show()

	# for i in range(5):
	# 	S = recuit_f1_p ( -5, 5,1, 10, 0.5 , 10000, 0.0001 , 5)
	# 	fig = plt.figure(i+1)
	# 	#plt.plot(S[0], S[1], 'k-x')
	# 	plt.plot(S1[0], S1[1], 'c-x')
	# 	plt.plot(X, f_1(X),  c='red')	
	# 	plt.xlabel('x')
	# 	#plt.title(strS)
	# 	plt.ylabel('f(x)')	
	



if Which_question==6:
	X = np.arange(-5,5,0.2)
	Y = np.arange(-5,5,0.2)
	X, Y = np.meshgrid(X, Y)
	Z= g(X,Y)


	S = recuit_g_p ( 0,0,0,0, 1, 15, 0.2 , 1000, 0.0001 , 5)
	S1 = recuit_g( 0,0,0,0, 1, 15, 0.2 , 1000, 0.0001 )

	fig = plt.figure(0)
	ax = fig.gca(projection='3d') #to perform a 3D plot
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
	
	ax.plot(S1[0],S1[1],S1[2], '-o',c='gray')
	ax.plot(S[0],S[1],S[2], '-o', c='cyan')
	ax.set_zlim(-300,300)
	ax.set_xlim(-5,5)
	ax.set_ylim(-5,5)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	
	
	print(S[1][-1])
	print(S1[1][-1])

	plt.show()


if Which_question==7:
	X = np.arange(-5,5,0.2)
	
	Y= f_1(X)


	S = recuit_f1_p ( 0, 0, 1, 10, 0.5 , 10000, 0.0001 , 5)
	S1 = recuit_f1( 0, 0 , 1, 10, 0.5 , 10000, 0.0001 )

	fig = plt.figure(0)

	plt.plot(S1[0], S1[1],  '-o',c='gray')
	plt.plot(S[0], S[1],  '-o', c='cyan')
	plt.ylim((-200,300))
	plt.xlim((-5,5))
	plt.plot(X, f_1(X),  c='black')	
	
	
	print(S[1][-1])
	print(S1[1][-1])

	plt.show()	

if Which_question==8 :
	X= np.arange(-6,6)
	F = min(f_1 (X))
	dist_max_sp =[]
	dist_max_ap = []
	Succes_sp = []
	Succes_ap = []
	for i in range(21):
		dist_sp = recuit_f1(-6, 6, 1,10, 0.5 , 10000, 0.0001 )
		f =max( abs(dist_sp[1]-  F) )
		dist_max_sp.append(f)
		if dist_sp[1][-1]-F<2 :
			Succes_sp.append(1)
		dist_ap = recuit_f1_p(-6, 6, 1,10, 0.5 , 10000, 0.0001,5)
		f2 =max( abs(dist_ap[1]-  F) )
		dist_max_ap.append(f2)
		if dist_ap[1][-1]-F<2 : 
			Succes_ap.append(1)

	print('distance max sans paliers', dist_max_sp)		
	print('distance max avec paliers', dist_max_ap)	
	print('nombre de succes sans paliers', Succes_sp)
	print('nombre de succe7s avec paliers', Succes_ap)











