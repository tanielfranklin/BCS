#import numpy as np

def normalizar_x(x,xnorm):
    return ((x - xnorm[:, 0]) / xnorm[:, 1])
def normalizar_u(u,unorm):
    return ((u - unorm[:, 0]) / unorm[:, 1])
def Fnorm(xlim):
    # Encontrar o fator de normalização
    # tal que xb=(x-x0)/xc
    # xmin<x<xmax
    # fazendo com que 0<xb<1
    return (xlim[0],xlim[1]-xlim[0])

def desnormalizar_x(x0,xnorm):
    return x0*xnorm[:, 1] + xnorm[:, 0]
def desnormalizar_u(x0,xnorm):
    return (x0.T*xnorm[:, 1] + xnorm[:, 0]).T
# def normaliza_u(ux,unorm):
#     aux=np.zeros((nu,1))
#     for i in range(0,len(ux)):
#        # aux[i]=(ux[i]-unorm[i,0])/unorm[i,1]
#         aux[i,0]=(ux[i,0]-unorm[i,0])/unorm[i,1]
#        # print(aux)
#     return aux
# def normaliza_y(y,ynorm):
#     aux=np.zeros((ny,1))
#     for i in range(0,len(y)):
#        # aux[i]=(ux[i]-unorm[i,0])/unorm[i,1]
#         aux[i,0]=(y[i,0]-ynorm[i,0])/ynorm[i,1]
#        # print(aux)
#     return aux
def desnormalizar_u(ux,unorm):
    aux=np.zeros((nu,1))
    for i in range(0,len(ux)):
       # aux[i]=(ux[i]-unorm[i,0])/unorm[i,1]
        aux[i,0]=ux[i,0]*unorm[i,1]+unorm[i,0]
       # print(aux)
    return aux

def desnormalizar_y(x,ynorm):
    aux=np.zeros((ny,1))
    for i in range(0,len(x)):
       # aux[i]=(ux[i]-unorm[i,0])/unorm[i,1]
        aux[i,0]=x[i,0]*ynorm[i,1]+ynorm[i,0]
       # print(aux)
    return aux
