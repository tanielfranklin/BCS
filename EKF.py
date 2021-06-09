#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 23:05:32 2021

@author: taniel
"""
import numpy as np
def EKF(process,xmk,yk,uk_1,W,V,Mk,ts,pv):
    ## Atualização da matriz A
    Ak = process['Linearizacao'](xmk,uk_1)[0];
    
    # EKF - Predicao
    [xpk,ypk] = process['Simulation'](xmk,uk_1);
    #print(ypk)
    # ymk = np.full((ypk.size1(),ypk.size1()), ypk);
    # xmk = np.full((xpk.size1(),xpk.size1()), xpk);
    ymk=ypk;
    xmk=xpk;
    # Calculo da matriz de covariancia Mk
    #print(Ak)
    #print(Ak.size1())
    #Phi = np.eye(len(xmk)) + np.full(Ak)*ts + (np.full(Ak)**2)*(ts**2)/2 + (np.full(Ak)**3)*(ts**3)/np.math.factorial(3); # Discretização
    Phi = np.eye(xmk.size1()) + Ak*ts + Ak**2*(ts**2)/2 + (Ak**3)*(ts**3)/np.math.factorial(3); # Discretização
    Mk = Phi*Mk*np.transpose(Phi) + W;
    
    # Linearizacao a cada k Atualização da matriz C
    Ck = process['Linearizacao'](xmk,uk_1)[2]; #pegar apenas a matriz C:
    
        
    Ck = Ck[pv,:]; # separando a matiz C só para as pv
    
    
    # EKF - correcao dos estados estimados
    Kf = Mk @ np.transpose(Ck) @ np.linalg.inv(Ck @ Mk @ np.transpose(Ck) + V);        # calculdo ganho
    
    
    Mk = (np.eye(xmk.size1()) - Kf@Ck)@Mk; # atualizacao da matriz de variancia dos estados estimados
    xmk = xmk + Kf@(yk - ymk); # correcao do estado
    
    ypk = process['Simulation'](xmk,uk_1)[1];
    ymk = ypk;
    return [xmk,ymk,Kf,Mk]
