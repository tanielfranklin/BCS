import numpy as np

class Dataset_Pinn(object):
    def __init__(self,dset,normx,normu,tsteps):
         self.remove_q=True        
         self.xc,self.x0=normx
         self.uc,self.u0=normu
         self.t_steps=tsteps
         self.N=dset[0].shape[0]
         self.data=self._dset_reshape(dset)
         self.xn=self._x_norm()
         self.un=self._u_norm()

    def _dset_reshape(self,dset):
        data=[np.reshape(i,(self.N,1)) for i in dset]
        return np.array(data).T[0]
    
    def _x_norm(self):
        val=[]
        x=self.data[:,-3:]  
        for i in range(3):
            val.append((x[:,i]-self.x0[i])/self.xc[i])
        return np.array(val).T
    def _u_norm(self):
        u=self.data[:,:4]
        val=[]
        for i,(j,k) in enumerate(zip(self.uc,self.u0)):
            val.append((u[:,i]-k)/j)
        un=np.array(val).T        
        return un

    def _split_sequences(self):
            #https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
        X, y, u = list(), list(),list()
        sequences=np.hstack((self.un,self.xn))
        n_steps_in, n_steps_out=self.t_steps
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out-1
            # check if we are beyond the dataset
            if out_end_ix > len(sequences)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y, seq_u= sequences[i:end_ix, :], sequences[end_ix:out_end_ix+1, :],sequences[end_ix-1:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        a=np.array(X)
        b=np.array(y)
        u_train=b[:,:,0:4]
        X_train=a[:,:,:]
        y_train=b[:,:,-3:]
            
        return X_train,y_train,u_train

    def _data_train(self):
        n_steps_in, n_steps_out = 30 ,10# convert into input/output
        a, b,c= self._split_sequences(self.data, n_steps_in, n_steps_out)
        X=a[:,:,:]
        y=b[:,:,-3:]
        u_train=b[:,:,0:4]

    def split_train_test(self):
        # Create train and test sets
        dset=self.data
        split_point = int(0.7*dset.shape[0]) # catch 70% for training
        X,y,u_train=self._split_sequences()
        train_X_full , train_y_full, u_train = X[:split_point, :] , y[:split_point, :], u_train[:split_point, :]
        test_X_full , test_y_full = X[split_point:, :] , y[split_point:, :]
        uk=dset[0:split_point,0:4]
        if self.remove_q==True:
            # Remove last state (q) unmeasured
            train_y=train_y_full[:,:,0:2]
            train_X=train_X_full[:,:,:-1]
            test_y=test_y_full[:,:,0:2]
            test_X=test_X_full[:,:,:-1]
        else:
            # Remove last state (q) unmeasured
            train_y=train_y_full
            train_X=train_X_full
            test_y=test_y_full
            test_X=test_X_full





            
        