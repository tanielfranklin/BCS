import numpy as np
import tensorflow as tf

class Dataset_Pinn(object):
    def __init__(self,dset,normx,normu,tsteps):
        self.remove_q=True        
        self.xc,self.x0=normx
        self.uc,self.u0=normu
        self.t_steps=tsteps
        self.N=dset[0].shape[0]
        self.data=self._dset_reshape(dset)
        self.xn=self._x_norm()
        self.un,self.u=self._u_norm()
        
        self.X,self.Y,self.U=self._DataSequence()
        self.Test=None
        n_features=6 # two network inputs  (fk, zc,pmc,prn, x1,x2)
        nu=4 #number of exogenous
        n_measured=2 # number of measured states
        self.dataset_LBFGS, self.dataset_ADAM = self._split_train_test()

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
        return un, np.array(u)

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
    def _split_sequences_u(self):
            #https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
        X, y, u = list(), list(),list()
        sequences=np.hstack((self.u,self.xn))
        n_steps_in, n_steps_out=self.t_steps
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out-1
            # check if we are beyond the dataset
            if out_end_ix > len(sequences)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y= sequences[i:end_ix, :], sequences[end_ix:out_end_ix+1, :]
            X.append(seq_x)
            y.append(seq_y)
        a=np.array(X)
        b=np.array(y)
        u_train=b[:,:,0:4]           
        return u_train

    def _DataSequence(self):
        n_steps_in, n_steps_out = self.t_steps# convert into input/output
        a, b,c= self._split_sequences()
        u_train=self._split_sequences_u()
        X=a[:,:,:]
        Y=b[:,:,-3:]
        U=b[:,:,0:4]
        return X,Y,u_train

    def _split_train_test(self):
        def ToTensor(x):
            return tf.convert_to_tensor(x, dtype=tf.float32)
        # Create train and test sets
        dset=self.data
        split_point = int(0.7*dset.shape[0]) # catch 70% for training
        train_X_full , train_y_full, u_train = self.X[:split_point, :] , ToTensor(self.Y[:split_point, :]), ToTensor(self.U[:split_point, :])
        test_X_full , test_y_full, self.u_test = self.X[split_point:, :] , self.Y[split_point:, :], ToTensor(self.U[split_point:, :])
        uk=ToTensor(dset[0:split_point,0:4])
        self.TestFull=[ToTensor(test_X_full),ToTensor(test_y_full)]
        if self.remove_q==True:
            # Remove last state (q) unmeasured
            train_y=ToTensor(train_y_full[:,:,0:2])
            train_X=ToTensor(train_X_full[:,:,:-1])
            self.Test=[ToTensor(test_X_full[:,:,:-1]),ToTensor(test_y_full[:,:,0:2])]
            
        else:
            # Remove last state (q) unmeasured
            train_y=ToTensor(train_y_full)
            train_X=ToTensor(train_X_full)
            self.Test=[ToTensor(test_X_full),ToTensor(test_y_full)]
            
        

           
        train_dataset = tf.data.Dataset.from_tensor_slices((train_X,train_y, u_train))
        batch_size=30
        train_dataset = train_dataset.batch(batch_size)
        train_X.shape
        data_LBFGS=[train_X,train_y,u_train]
        data_ADAM = train_dataset
        return data_LBFGS, data_ADAM
        
        
        





            
        