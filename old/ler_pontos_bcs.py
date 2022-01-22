import numpy as np
import matplotlib.pyplot as plt
exec(compile(open('param.py', "rb").read(), 'param.py', 'exec')) #% Roda arquivo com modelo BCS
F1c=941799.5331
F2c=2260318.8795
qcc=0.033987702
Hc=1511.97
tc=1/(PI*b1/V1)#tc=1
pbc=pr
qc=pbc*PI#;qc=1/30
pwc=tc*b2*qc/V2
data = np.load('BCS_data_train.npz')
x1=data['x1']/pbc
x2=data['x2']/pwc
x3=data['x3']/qc
tempo=data['t']#/tc
tempo=tempo.reshape(len(x1),1)
print('x1.shape'+str(x1.shape))
print('x2.shape'+str(x2.shape))
print('x3.shape'+str(x3.shape))
print('tempo.shape'+str(tempo.shape))
tempo=tempo/tc
print(tc)
fig3=plt.figure()
label = ['Pbh','Pwh','q'];
ax3=fig3.add_subplot(3,1,1)
ax3.plot(tempo,x1, label='Medição')
ax3.set_ylabel(label[0])
ax3.grid()
ax3=fig3.add_subplot(3,1,2)
ax3.plot(tempo,x2, label='Medição')
ax3.set_ylabel(label[1])
ax3.grid()
ax3=fig3.add_subplot(3,1,3)
ax3.plot(tempo,x3, label='Medição')
ax3.set_ylabel(label[2])
ax3.grid()
plt.show()
