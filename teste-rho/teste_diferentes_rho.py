import matplotlib.pyplot as plt
import numpy as np

## Carregar Exógena externa


fig3=plt.figure()
label = ['Pbh (bar)','Pwh (bar)','q(m3/s)'];
dados = np.load('teste-rho/BCS_data_train_aprbs_f-1.npz')
t=dados['t']


x1=np.zeros((4,100))
x2=np.zeros_like(x1)
x3=np.zeros_like(x1)
for i in range(1,5):
    dados = np.load('teste-rho/BCS_data_train_aprbs_f-'+str(i)+'.npz')
    x1[i-1,:]=dados['x1'].T
    x2[i-1,:]=dados['x2'].T
    x3[i-1,:]=dados['x3'].T


print(x1.shape)
label_rho = ['950','900','850','800'];
color=['0.8','0.7','0.6','0.5']
color=np.arange(0.2,1,0.2)
print(color)
c=[];
for j in np.arange(0,len(color)):
    c.append(str(color[j]))
print(c)

for iu in range(0,3):
    ax3=fig3.add_subplot(3,1,iu+1)
    if iu==2:
        for j in range(0,4):
            ax3.plot(t,x3[j,:]*3600,c[j], label=label_rho[j])
        ax3.set_ylabel(label[iu])
        plt.grid(True)
    elif iu==0:
        for j in range(0,4):
            ax3.plot(t,x1[j,:]/1e5,c[j], label=label_rho[j])
        ax3.set_ylabel(label[iu])
        plt.grid(True)
    else:
        for j in range(0,4):
            ax3.plot(t,x2[j,:]/1e5,c[j], label=label_rho[j])
        ax3.set_ylabel(label[iu])
        plt.grid(True)
plt.legend()
plt.show()

