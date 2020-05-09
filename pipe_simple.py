# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:59:21 2020

@author: Gess
"""


#!/usr/bin/python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import custom_plot as cplt
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

im = Image.open('channels.jpg') 
pix = im.load()
m, mm = im.size  # Get the width and height of the image for iterating over
pxs = list(im.getdata())
pxs = [pxs[i * mm:(i + 1) * mm] for i in range(m)]

H2 = []
H3 = []
f=open("channels_2.txt","r")
for li in f:
    K  = []
    for ch in li:
        if ch == '1':
            H3.append(int(ch))
        elif ch == '0':
            H3.append(int(ch))
         #m*height(j) + width(k) 
    # Read in the (x,y) positions of walls and empty spaces
    #b=li.split()
        
        if ch == '1':
            K.append(int(ch))
        elif ch == '0':
            K.append(int(ch))

    #for B in b:
     #   fp.append(int(B))
    H2.append(K)  

f.close()  

#H = H3.copy()
#H = H[0:228]


# Grid setup
n = 4
u0 = 1
m = 114
mm = m
h=1.0/(m+1)
uu=np.zeros((m+2,mm+2))
vv=np.zeros((m+2,mm+2))
u=np.zeros((n,m*mm))   #V_x
v=np.zeros((n,m*mm))  #V_y
speed  = np.zeros((m+2,mm+2)) 
ww=np.zeros((m+2,mm+2))
psii=np.zeros((m+2,mm+2))
w = np.zeros((n,m*mm))
psi= np.zeros((n,m*mm))
# Create derivative matrix and source term
d=np.zeros((mm*m,mm*m))
f=np.zeros((mm*m))
tol = np.ones((n,1))
hfac=1/(h*h)

for k in range(n):
    if k>0:
        for j in range(0,m):
            for i in range(0,m): 
                
                ij=i+m*j
                indices = [i-1+m*j,(j-1)*m+i]
                neighborz = [psi[k-1][(j-1)*m+i],psi[k-1][j*m+i-1]]
                if i > m-2:
                    neighborz.append(psi[k-1][i+m*j])
                else:
                    neighborz.append(psi[k-1][i+m*j+1])
                if j > m-2:
                    neighborz.append(0)
                else:
                    neighborz.append(psi[k-1][i+m*(j+1)])
                for l in range(2):
                    if indices[l] < 0:
                        neighborz[l] = 0
    #                    print(p[i][j][k])  
                for l in range(4):
                    w[k][i+m*j] += neighborz[l]/h**2
                    #p[i+1][j][k] = (nu)*(p[i][j+1][k]+p[i][(j-1)][k]-4*p[i][(j)][k]+p[i][(j)][k-1]+p[i][(j)][k+1])+p[i][(j)][k]
                
                w[k][i+m*j] += 4*psi[k-1][i+m*j]/h**2
    if k ==0:    
        # for i in range(m):
        #     for j in range(m):
        #         ij=i+m*j
        #         if i == 0 or i == m-1:
        #             d[ij,ij]=1
        #             f[ij] = 0
        
        #         # Derivative matrix
        #         else:    
        #             d[ij,ij]=-4*hfac
        #             if i>0: d[ij,ij-1]=hfac
        #             if i<m-1: d[ij,ij+1]=hfac
        #             if j>0: d[ij,ij-m]=hfac
        #             if j<m-1: d[ij,ij+m]=hfac
        
        #         # Source term
        
        #         f[ij]=0
                
        # w[k]=np.linalg.solve(d,f)
        wsolv = w[k].copy()
    wsolv = w[k].copy()       
    for i in range(m):
        for j in range(m):
            ij=i+m*j
            if i == 0 or i == m-1:
                d[ij,ij]=1
                wsolv[ij] = 1
            if j == 0 or j ==m-1:
                d[ij,ij]=1
                wsolv[ij] = 0
            # Derivative matrix
            else:    
                d[ij,ij]=-4*hfac
                if i>0: d[ij,ij-1]=hfac
                if i<m-1: d[ij,ij+1]=hfac
                if j>0: d[ij,ij-m]=hfac
                if j<m-1: d[ij,ij+m]=hfac
    
            # Source term
    
            
    psi[k]=np.linalg.solve(d,wsolv)
    for i in range(m-2):
        for j in range(mm-2):
            ij=i+m*j
            v[k][ij] = -(psi[k][i+m*j+1] - psi[k][i+m*j-1])/(2*h) #centered difference
            u[k][ij] = (psi[k][i+m*(j+1)] - psi[k][i+m*(j-1)])/(2*h)
            tol[k] = np.linalg.norm(np.add(u[k][ij],-u[k-1][ij]))
    if tol[k] < 1e-3:
        break

# Reconstruct full grid
for i in range(m):
    ww[i+1,1:mm+1]=w[n-1][i*m:(i+1)*m]
    psii[i+1,1:mm+1]=psi[n-1][i*m:(i+1)*m]
    uu[i+1,1:mm+1]=u[n-1][i*m:(i+1)*m]
    vv[i+1,1:mm+1]=v[n-1][i*m:(i+1)*m]
    for j in range(m):
        speed[j][i] = np.sqrt(uu[j][i]**2 + vv[j][i]**2)

# for j in range(1,mm):
#     for i in range(1,m):
#         # vv[j][i] = -(psii[j][i+1] - psii[j][i-1])/(2*h) #centered difference
#         # uu[j][i] = (psii[j+1][i] - psii[j-1][i])/(2*h)
#         speed[j][i] = np.sqrt(u[j][i]**2 + v[j][i]**2)
    

# Plot using Matplotlib
# xa=np.linspace(0,1,m+2)
# mgx,mgy=np.meshgrid(xa,xa);
# fig=plt.figure()
# ax=fig.gca(projection='3d')
# surf=ax.plot_surface(mgx,mgy,uu,rstride=1,cstride=1,linewidth=0)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()
start, stop, n_values = 0, m+2,m+2
# step = np.max(ww)/10000
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(0,mm+2,mm+2 )
X, Y = np.meshgrid(x_vals, y_vals)
# levels = np.arange(0.0, np.max(ww), step) + step
#plt.contourf(X, Y, uu, levels, alpha=1,cmap=cm.Blues)
lw = 5*speed / speed.max()
cp1=plt.contourf(X, Y, speed, 16,cmap=cm.jet)
cp2=plt.contour(X, Y, psii, 16, colors='black', linewidth=.5)
plt.colorbar(cp1)
plt.show()
cp3 = plt.streamplot(X,Y,vv,uu, density=0.6, color='k', linewidth=lw)

