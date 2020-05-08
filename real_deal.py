# -*- coding: utf-8 -*-
"""
Created on Sun May  3 09:16:15 2020

@author: Gess
"""

import numpy as np
import custom_plot as cplt
import math
import matplotlib.pyplot as plt
from matplotlib import cm
#from hw52_floor import fp
from PIL import Image


#import the image of the channels 
im = Image.open('channels.jpg') 
pix = im.load()
m, mm = im.size  # Get the width and hight of the image for iterating over
pxs = list(im.getdata())
pxs = [pxs[i * mm:(i + 1) * mm] for i in range(m)] 

H2 = []
H3 = []

#read the channels file and store the location of the walls
f=open("channels_2.txt","r")
for li in f:
    K  = []
    for ch in li:
       #m*height(j) + width(k) 
    # Read in the (x,y) positions of walls and empty spaces
    #b=li.split()
        
        if ch == '1':
            K.append(int(ch))
            H3.append(int(ch))
        elif ch == '0':
            K.append(int(ch))
            H3.append(int(ch))
    #for B in b:
     #   fp.append(int(B))
    H2.append(K)  

f.close()    






# Grid size and time steps
n = 10
dt = 0.24
#n=100
v_x=np.zeros((n,mm,m)) #2d assumption
v_x0 = np.ones((1,m))

v_x[0][0] = v_x0
v_x[0] = np.transpose(v_x[0])

v_y=np.zeros((n,mm,m)) #initially 0, only in x direction
p = [H2.copy() for i in range(n)]

#dpdx = 1 #constant pressure gradient, only dependent on x

# given constants
c=1 
h= 1
eta = 1 #viscosity rho*nu
dt = 0.24
#dt = 0 #assumption1: fully formed profile
nu=dt/(h**2)
rho = 1
#rho*dv_xdt-eta*Deltav_x=-dpdx



#Delta v_x = 
   #dvxdx+dvydy = 0 dvydy = - dvxdx 


   
#Integrate
t=0

w = np.zeros((mm,m))

#Delta omega  = 0
for j in range(0,len(H2)):
   w[j][0] = 1
   w[j][len(H2[0])-1]=1
   for k in range(1,len(H2[0])-1):
       if H2[j][k] == 1:
            w[j][k] = 0
            continue
       indices = [k-1,j-1,j+1,j-1]
       neighborz = [w[j-1][k],w[j][k-1],w[j][k+1]]
       if j < mm-1:
           neighborz.append(w[j+1][k])
           H2_ind = [H2[j-1][k],H2[j][k-1], H2[j][k+1],H2[j+1][k]]
           for l in range(4):         
               if H2_ind[l] == 1:
                   neighborz[l] = 0
               if l <2:
                  if indices[l] < 0:
                      neighborz[l] = 0 
       else:
           neighborz.append(0)
       for l in range(4):
             #??
#                    print(p[i][j][k])
        
           w[j][k] += neighborz[l]/(4*h**2)
                #p[i+1][j][k] = (nu)*(p[i][j+1][k]+p[i][(j-1)][k]-4*p[i][(j)][k]+p[i][(j)][k-1]+p[i][(j)][k+1])+p[i][(j)][k]
          
          #  v_x[i+1][j][k] += v_x[i][j][k] - 4*nu*v_x[i][j][k] - dpdx
            # if abs(p[i+1][91][188]) > 10e-3:











# for i in range(0,n-1):
#     t += dt
#     print(t)   #counter
    
#     for j in range(0,len(H2)):
#         v_x[i][j][0] == 1
#         for k in range(1,len(H2[0])-1):
#             indices = [k-1,j-1,j+1,j-1]
            
#             neighborz = [v_x[i][j-1][k],v_x[i][j][k-1]]
#             dpdx = (p[i][j][k] - p[i][j][k-1])/h
#             if k > m-2:
#            #     neighborz.append(v_x[i][j][k])
#                 neighborz.append(0)
                
#             else:
#                 neighborz.append(v_x[i][j][k+1])
#             if j > mm-2:
#       #          neighborz.append(v_x[i][j][k])
#                 neighborz.append(0)
#             else:
#                 neighborz.append(v_x[i][j+1][k])
#             for l in range(2):
#                 if indices[l] < 0:
#                     neighborz[l] = v_x[i][j][k]  #??
# #                    print(p[i][j][k])
            
            
#             try:
#                 for l in range(4):
#                     H2_ind = [H2[j-1][k],H2[j+1][k],H2[j][k-1], H2[j][k+1]]
#                     if H2_ind[l] == 1:
#                         neighborz[l] = 0
#             except LookupError:
#                 pass
#                 v_x[i+1][j][k] += nu*neighborz[l]
#                 #p[i+1][j][k] = (nu)*(p[i][j+1][k]+p[i][(j-1)][k]-4*p[i][(j)][k]+p[i][(j)][k-1]+p[i][(j)][k+1])+p[i][(j)][k]
#             if H2[j][k] == 1:
#                 v_x[i+1][j][k] = 0
#             v_x[i+1][j][k] += v_x[i][j][k] - 4*nu*v_x[i][j][k] - dpdx
 
q=np.loadtxt("ch114.txt",dtype=np.int8)
#cplt.plot2("w1.pdf",w,q,-1.1,1.1,3)    
start, stop, n_values = 0, m,m
step = np.max(w)/1000
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(0,mm,mm )
X, Y = np.meshgrid(x_vals, y_vals)
levels = np.arange(0.0, np.max(w), step) + step
plt.contourf(X, Y, w, levels, alpha=1,cmap=cm.Blues)
#np.savetxt('data.csv', data, delimiter=',')    