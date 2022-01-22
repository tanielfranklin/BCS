import matplotlib.pyplot as plt
import numpy as np
exec(compile(open('param.py', "rb").read(), 'param.py', 'exec')) #% Roda arquivo com parâmetros do modelo BCS
exec(compile(open('subrotinas.py', "rb").read(), 'subrotinas.py', 'exec')) #% Roda arquivo com parâmetros do modelo BCS
a_range = [35,65] # intervalo de variaçãdo nível
b_range = [10,15] # duração em cada nível
tsim=100;ts=1
nstep = tsim
nsim=tsim/ts
dudtmax = np.array([0.5, 1])

u_f,prbs=APRBS(a_range,b_range,nstep) # variação em f
z_f,prbs=APRBS([35,100],b_range,nstep)
uk_1 = np.array([u_f,np.ones_like(u_f)*60]);
uk_1 = np.array([np.ones_like(u_f)*80,np.ones_like(u_f)*100]);
xi=(np.arange(0,int(nsim*ts),ts));

dfq_max = 0.5;    # m�xima varia��o em f/s
dzc_max = 1;  # m�xima varia��o em zc #/s
tp =np.array([[1/dfq_max,1/dzc_max]]).T;
print(uk_1.shape)
# Restricao do Elemento Final
def input_bound(u,dudtmax,tp):
    uref=[50,50]
    ur=np.zeros_like(u)
    for i in np.arange(0,len(u[1,:])):
        
        df=abs((u[0,i]-uref[0])/tp[0])
        df=abs((u[0,i]-uref[0])/ts)
        dz=abs((u[1,i]-uref[1])/tp[1])
        dz=abs((u[1,i]-uref[1])/ts)
        print([df,dz])
        if df>dudtmax[0]:
            ur[0,i]=np.sign(df)*dudtmax[0]
            ur[0,i]=uref[0]-ts*dudtmax[0]
        else:
            ur[0,i]=u[0,i]
            
        if dz>dudtmax[1]:
            ur[1,i]=np.sign(dz)*dudtmax[1]
            ur[1,i]=uref[1]-ts*dudtmax[1]
        else:
            ur[1,i]=u[1,i]
        uref=u[:,i]
        print('uref: ',uref, 'ur: ', ur[:,i])
        
    return ur

u=input_bound(uk_1,dudtmax,tp)
print('uk',uk_1.shape)
print('u',u.shape)
print('xi',xi.shape)
fig=plt.figure()
ax2=fig.add_subplot(211)
ax2.plot(xi,u[0,:])
ax2.plot(xi,u[1,:])
ax2=fig.add_subplot(212)
ax2.plot(xi,uk_1[0,:])
ax2.plot(xi,uk_1[1,:])
ax2.set_ylabel('rho')
plt.show()