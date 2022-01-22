import time
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
import pandas as pd
from pandas import Series
from pandas import concat
#from pickle import dump
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.callbacks import ModelCheckpoint
from math import sqrt
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate as interp

def gen_traindata(file):
	data = np.load(file)
	return data["t"], data["x"], data["u"]
# time points

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value[0]] # concatena uma lista de	X com yhat (value)
	array = np.array(new_row) # converte para array
	array = array.reshape(1, len(array)) # Faz a transposta
	inverted = scaler.inverse_transform(array)	# reescala
	return inverted[0, -1] # retorna yhat (value) reescalonado

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

def forecast_on_batch(model, batch_size, X):
	X = X.reshape(batch_size, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

exec(compile(open('exp_LEA.py', "rb").read(), 'exp_LEA.py', 'exec'))
exec(compile(open('param_LEA.py', "rb").read(),'param_LEA.py', 'exec'))
exec(compile(open('subrotinas.py', "rb").read(), 'subrotinas.py', 'exec'))

# =========================================================================
#  Define as entradas do BCS LEA   
# =========================================================================
#   freq = Cexp.referencia_frequencia_inversor(1:nsim)*0.1;                   % [Hz] frequencia de operacao
intervalo=np.array([int(0*3600),int(7.9*3600)])

#intervalo=np.array([1.8,3])
LEA=dados_LEA_Exp('Dados_BCSLEA_20210818.mat',intervalo)
pm=LEA['pressao_manifold_coriolis']*1e5
pr=LEA['pressao_reservatorio']*1e5
pm_0=pm[0]
pr_0=pr[0]
u_0=np.array([LEA['referencia_frequencia_inversor'][0],LEA['valvula_pneumatica_topo'][0],pm_0,pr_0])
x_0=np.array([LEA['pressao_fundo'][0]*1e5,LEA['pressao_choke'][0]*1e5,LEA['vazao'][0]/3600])
u_0
nsim=LEA['tempo']+1
ts=LEA['Ts']
tempo_hora = np.arange(0,nsim*ts,ts)/3600

#========================
# Filtragem Pman=========
fs=1/ts
fs
Wn=2*pi*fs/1.2
input_signal=np.reshape(pm,(1,nsim))
b, a = sp.signal.butter(2, Wn, 'low')
output_signal = np.reshape(sp.signal.filtfilt(b, a, input_signal),(nsim))

fig3=plt.figure()
label = ['Pman','Pr','f','z'];
ax1=fig3.add_subplot(2,2,1)
ax1.plot(tempo_hora ,pm/1e5, label='Pman')
ax1.plot(tempo_hora ,output_signal/1e5, ':r')
ax1.set_ylabel(label[0])
plt.grid(True)
ax2=fig3.add_subplot(2,2,3)
ax2.plot(tempo_hora,pr/1e5, label='Pr')
ax2.set_ylabel(label[1])
plt.grid(True)
ax3=fig3.add_subplot(2,2,2)
ax3.plot(tempo_hora,LEA['referencia_frequencia_inversor'], label='f')
ax3.set_ylabel(label[2])
plt.grid(True)
ax4=fig3.add_subplot(2,2,4)
ax4.plot(tempo_hora,LEA['valvula_pneumatica_topo'], label='z')
ax4.set_ylabel(label[3])
plt.grid(True)
#plt.show()

fk=LEA['referencia_frequencia_inversor']
zc=LEA['valvula_pneumatica_topo']

# Entradas
# Valores máximos e mínimos para normalização
#Entradas - conforme binder e pavlov
#========================================
pbc=Lim_c(pbhlim)
pwc=Lim_c(pwhlim)
qc=Lim_c(qlim)
pbmin=pbhlim[0]
pwmin=pwhlim[0]
qmin=qlim[0]


rho=tf.Variable(8.18)*100 #836.8898
PI = tf.Variable(2.5)*1e-8 ##PI = 2.7e-8; # Well productivy index [m3/s/Pa]
xc=np.array([pbc,pwc,qc])
x0=np.array([pbmin,pwmin,qmin])
x_0=np.array([LEA['pressao_fundo'][0]*1e5,LEA['pressao_choke'][0]*1e5,LEA['vazao'][0]/3600])
x1=(LEA['pressao_fundo']*1e5-x0[0])/xc[0]
x2=(LEA['pressao_choke']*1e5-x0[1])/xc[1]
x3=(LEA['vazao']/3600-x0[2])/xc[2]
x1=x1.reshape(nsim,1)
x2=x2.reshape(nsim,1)
x3=x3.reshape(nsim,1)
tempo=np.arange(0,nsim)
tempo=tempo.reshape(len(x1),1)

F1c=2.92634e-05
F2c=0.000738599
Hc=215.9226497597185
qcc=0.0020328441729756536
F1lim=(0.000439365,0.000439365)
F2lim=(0.0110894,0.0110894)

u=np.hstack([fk.reshape(len(fk),1),zc.reshape(len(fk),1),pm.reshape(len(fk),1),pr.reshape(len(fk),1)])
un=np.hstack([fk.reshape(len(fk),1)/60,zc.reshape(len(fk),1)/100,1e-2*np.sqrt(np.abs(x2-pm.reshape(len(fk),1))),1e-5*pr.reshape(len(fk),1)-x1])
#np.hstack([x1,x2,x3,u])[0:2]
#df1 = DataFrame(dados)

#df = DataFrame(np.hstack([x,u.reshape(len(u),1),t.reshape(len(t),1)]),columns=['y', 'u','t'])
#dff = DataFrame(np.array([[0],[1],[2]]),columns=['x11','x22','x33'])
df = pd.DataFrame(np.hstack([x1,x2,x3,un]),columns=['pbh','pwh','q','fn','zn','pmn','prn'])
df_u = pd.DataFrame(np.hstack([u]),columns=['f','z','pm','pr'])
# Defining a batch size based on the data
batch_size=15
# Split the dataset into different batches
batch_data = np.array_split(df, int(df.shape[0]/batch_size))
nvar=len(df.columns)
print(df.head())
print(df_u.head())

dset = df.values.astype(float)
du_set = df_u.values.astype(float)
N_star=4*batch_size # test size
look_back=1
Ns=3 # Number of states
Nu=4 # Number of inputs


# Create train and test sets
train_set = dset[:-N_star,:]
test_set = dset[-N_star:,:]
u_set=du_set[:-N_star,:]
N=train_set.shape[0]

print(test_set.shape)

supervised_train = timeseries_to_supervised(train_set, look_back)
supervised_test = timeseries_to_supervised(test_set, look_back)

train = supervised_train.values#[:,0:(Ns+Nu+Ns)]
test = supervised_test.values#[:,0:nvar*look_back+2]
# train = train_norm
# test = test_norm
print("Test data head")
print(test_set[0:3])
print("==========================")
print("Supervised test data head")
print(test[0:3])
("==========================")

print('Test:')
print('x(k-1)===========u(k-1)========x(k)========u(k)')
print(test[0:2])

x_train=tf.convert_to_tensor(train[:,0:Ns+Nu], dtype=tf.float32)
y_train=tf.convert_to_tensor(train[:,Ns+Nu:Ns+Nu+Ns], dtype=tf.float32)
yk1=tf.convert_to_tensor(train[:,0:Ns], dtype=tf.float32) # y(k-1) para ODE
uk=tf.convert_to_tensor(u_set, dtype=tf.float32) # u(k) para ODE

#LSTM layer inputs require x_train with 3 dimensions: first dataset lines (batch), second timestep, NN inputs number
x_train=tf.reshape(x_train, [x_train.shape[0],1,Ns+Nu])
# Splitting in batches
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, tf.concat([y_train,yk1],axis=1),yk1,uk))
train_dataset = train_dataset.batch(batch_size)

def ED_BCS(x,xk,u):
    ## Montado o sistema de equa��es
    # Tensores (Estados atuais preditos)
    pbh = x[:,0:1]
    pwh = x[:,1:2]
    q = x[:,2:] #Vazão
    #Valores passados de x
    pbhk = xk[:,0:1]
    pwhk = xk[:,1:2]
    qk = xk[:,2:] #Vazão

    #Entradas exógenas atuais
    fq=u[:,0:1]
    zc=u[:,1:2]
    pman=u[:,2:3];#pman=0.12e5
    pres=u[:,3:4]

    #=============================================
    # Computing HEAD and pump pressure gain of LEA
    q0 = (qc*q+qmin) / Cq * (f0 / fq)
    H0 =  Head[0]*q0**4 +  Head[1]*q0**3 +  Head[2]*q0**2 + Head[3]*q0 + Head[4];
    H = CH * H0 * (fq / f0) ** 2  # Head
    #Pp = rho * g * H  # Dp
    #==============================================
    # Electrical power and electrical current computing
    #P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4
    #P = Cp * P0 * (fq / f0) ** 3;  
    #I = Inp * P / Pnp  
    #==============================================
    # Computing two volumes frictions in LEA piping
    qan=q*qc+qmin # non normalized flow
    Re =(4*rho*qan)/(0.219*pi*mu); # Assuming volumes density are identicals
    fric=64/Re 
    
    F1 = (fric*qan**2*rho)/(2*pi*r1**3) #Frictional pressure drop above ESP (Assuming laminar flow)
    F2 = (fric*qan**2*rho)/(2*pi*r2**3) #Frictional pressure drop above ESP (Assuming laminar flow)
    #===========================================
    #===========================================
    # Computing intake pressure
    #pin = pbh*pbc+pbmin - rho * g * h1 - F1;
    # Computing Reservoir flow
    qr = PI * (pres - (pbh*pbc+pbmin));
    # Computing flow across Choke valvule
    qch = Cc * (zc) * tf.sqrt(tf.abs(pwh*pwc+pwmin - pman)); # Algumas operações precisam usar o prefixo do tensorflow 
    #============================================

    #Normalizing nonlinear terms
    ##########################
    qch=(qch-qch_lim[0])/qcc
    F1=(F1-F1lim[0])/F1c
    F2=(F2-F2lim[0])/F2c
    H=(H-H_lim[0])/Hc
    ###########################
    #xss=np.float32(np.array([2.0197e5,4.9338e5,4.2961e-4]));

    # SEDO

    return tf.stack((
        (pbh-pbhk)/ts - (1/pbc)*b1/V1*(qr - (q*qc+qmin)),
        (pwh-pwhk)/ts - (1/pwc)*b2/V2*((q*qc+qmin) - (qcc*qch+qch_lim[0])),
        (q-qk)/ts - (1/(qc*M))*(pbh*pbc+pbmin - (pwh*pwc+pwmin) - rho*g*hw - (F1c*F1+F1lim[0]) - (F2c*F2+F2lim[0]) + rho * g * (H*Hc+H_lim[0]))
    ))
print("=============================")
print("yk")
print(y_train[0:2,0:3].numpy())
print("=============================")
print("yk")
print(yk1[0:2,0:3].numpy())
print("=============================")
print("uk")
print(uk[0:2,0:3].numpy())
print("=============================")
print(ED_BCS(y_train[1:30,0:3],yk1[1:30],uk[1:30]).numpy())
print(ED_BCS(y_train[1:2,0:3],yk1[1:2],uk[1:2]).numpy()**2)
print(np.sqrt(ED_BCS(y_train[1:2,0:3],yk1[1:2],uk[1:2]).numpy()**2))
print(np.mean(np.sqrt(ED_BCS(y_train[1:2,0:3],yk1[1:2],uk[1:2]).numpy()**2)))
print(np.mean(np.sqrt(ED_BCS(y_train[1:,0:3],yk1[1:],uk[1:]).numpy()**2)))

