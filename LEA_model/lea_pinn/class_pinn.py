# from lea_pinn import ODE_LEA
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from lea_pinn import TrainState,LossHistory,VarHistory#,get_abs_mean_grad,get_abs_max_grad
from lea_setup import *
import numpy as np


class PhysicsInformedNN(object):
    #Define the Constructor
    def __init__(self,neurons,tsteps, optimizer, logger,  var=None, pinn_mode=1, inputs=2):
    #N_in,N_out define time-steps in and out of the network
    # Descriptive Keras model LSTM model
        N_in,N_out=tsteps
        self.u_model = Sequential()
        self.var=[1,1]
        
        self.inputs=inputs #input states
        n_features=inputs
        # encoder layer
        self.u_model.add(LSTM(neurons, input_shape=(N_in,inputs)))
        self.u_model.add(Dropout(0.2))
        self.u_model.add(RepeatVector(N_out))
        # decoder layer
        self.u_model.add(LSTM(neurons, return_sequences=True))
        self.u_model.add(Dropout(0.2))  
        self.u_model.add(TimeDistributed(Dense(3)))#,input_shape=(n_steps_out, 2)))
        
        output_start=self.u_model.predict(tf.random.normal(shape=(1,N_in,inputs),dtype=tf.float32)) # forçar o inicio dos parâmetros do modelo
        self.train_state = TrainState()

        self.bestWeights=self.u_model.get_weights()
        self.bestLoss=np.inf
        self.optimizer = optimizer   
        self.logger = logger
        self.dtype = tf.float32
        # self.test_X,self.test_y=data_test
        
        #self.Loss_Weight_pinn=loss_weight_pinn
        self.alfa=tf.constant(0.8, dtype=tf.float32)
        self.lamb_bc=tf.constant(1.0, dtype=tf.float32)
        self.lamb_l1=tf.constant(1.0, dtype=tf.float32)
        self.lamb_l2=tf.constant(1.0, dtype=tf.float32)
        self.lamb_l3=tf.constant(1.0, dtype=tf.float32)
        self.pinn_mode=1
        self.ODE=ODE_LEA
    
        self.pinn_mode_set(pinn_mode)

        self.loss_new=True
        if self.pinn_mode!=0:
            self.rho = tf.Variable(var[0], dtype=tf.float32)
            self.PI = tf.Variable(var[1], dtype=tf.float32)
        else:
            self.rho=tf.Variable(1.0, dtype=tf.float32)
            self.PI=tf.Variable(1.0, dtype=tf.float32)



        # self.error_fn=self.erro()
        self.Font=14   
        # self.logger.set_error_fn(self.erro)
        self.losshistory = LossHistory() 
        self.varhistory = VarHistory()
        self.nsess=0 
        self.epoch=0
        self.logger.log_train_start(self)
    
    def get_lamb_weights(self):   
        l1=f"[{self.lamb_bc.numpy():4.3f},{self.lamb_l1.numpy():4.3f},{self.lamb_l2.numpy():4.3f},{self.lamb_l3.numpy():4.3f},"
        #l2=f"{self.lamb_l1.numpy():4.3f},{self.lamb_l2.numpy():4.3f},{self.lamb_l3.numpy():4.3f}]"
        return l1#+l2


    def pinn_mode_set(self,value):
        cases = {
            # Turn off all loss terms linked with EDO
            "off": lambda: 0,
            # Turn off all loss terms linked with EDO
            "on": lambda: 1,
            # Turn on all loss terms linked with EDO
            "all": lambda: 2,
            # Turn off main loss_EDO and keep loss EDO2
            "loss2": lambda: 3,
        }
        if value in cases.keys():        
            self.pinn_mode=cases[value]()
        else:
            raise ValueError("Invalid arguments for pinn_mode")

    def function_factory(self, train_x, train_y,uk,dataset):
        #function used to L-BFGS Adapted from Pi-Yueh Chuang <pychuang@gwu.edu>
        # Copyright © 2019 Pi-Yueh Chuang <pychuang@gwu.edu>
        #
        # Distributed under terms of the MIT license.

        """An example of using tfp.optimizer.lbfgs_minimize to optimize a TensorFlow model.

        This code shows a naive way to wrap a tf.keras.Model and optimize it with the L-BFGS
        optimizer from TensorFlow Probability.

        Python interpreter version: 3.6.9
        TensorFlow version: 2.0.0
        TensorFlow Probability version: 0.8.0
        NumPy version: 1.17.2
        Matplotlib version: 3.1.1
        """
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

        Args:
            model [in]: an instance of `tf.keras.Model` or its subclasses.
            loss [in]: a function with signature loss_value = loss(pred_y, true_y).
            train_x [in]: the input part of training data.
            train_y [in]: the output part of training data.

        Returns:
            A function that has a signature of:
                loss_value, gradients = f(model_parameters).
        """

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.wrap_training_variables())
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = [] # stitch indices
        part = [] # partition indices

        for i, shape in enumerate(shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n

        part = tf.constant(part)

        #@tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.
            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """
            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.wrap_training_variables()[i].assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        #@tf.function
        def f(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.

            This function is created by function_factory.

            Args:
            params_1d [in]: a 1D tf.Tensor.

            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """
            rho = 836.8898;
            PI = 2.7e-8;

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                #loss_value = self.myloss(self.u_model(train_x, training=True), train_y)
                #loss_value = self.myloss(self.u_model(train_x), train_y)
                loss_bc,loss_x1,loss_x2,loss_x3,loss_f= self.GetLoss(train_y, self.u_model(train_x),uk)   
                loss_f=self.lamb_l1*loss_x1+self.lamb_l2*loss_x2+self.lamb_l3*loss_x3
                #loss_f=loss_x1+loss_x2+loss_x3
                loss_value=loss_bc+loss_f
  

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, self.wrap_training_variables())
            grads = tf.dynamic_stitch(idx, grads)
            

            # print out iteration & loss
            f.iter.assign_add(1)
            self.train_state.step=str(f.iter.numpy())
            self.train_state.rho=self.rho*rho
            self.train_state.PI=self.PI*PI
            self.train_state.loss_test=self.erro(dataset.Test).numpy()
            self.train_state.loss_train=loss_value.numpy()
            self.train_state.loss_train_bc=self.lamb_bc*loss_bc.numpy()
            self.train_state.loss_train_f=loss_f.numpy()
            self.train_state.loss_train_x1=self.lamb_l1*loss_x1.numpy()
            self.train_state.loss_train_x2=self.lamb_l2*loss_x2.numpy()
            self.train_state.loss_train_x3=self.lamb_l3*loss_x3.numpy()
            self.train_state.weights=self.u_model.get_weights()
            self.train_state.update_best() 
            if f.iter%10==0:
                #tf.print("Iter:", f.iter, loss:{loss_value:.4e}")
                #tf.print("Iter:", f.iter, "loss:", loss_value)
                custom_log_res=f"[{loss_x1.numpy()*self.lamb_l1.numpy():.2e},{loss_x2.numpy()*self.lamb_l2.numpy():.2e},{loss_x3.numpy()*self.lamb_l3.numpy():.2e}]"
                custom_log=f"{self.rho.numpy()*rho:.1f} {self.PI.numpy()*PI:.2e}"
                #custom_log=f" lambda=[{self.lamb_l1:.1f},{self.lamb_l2:.1f},{self.lamb_l3:.1f}], rho={self.rho.numpy()*rho:.1f}, PI={self.PI.numpy()*PI:.2e}"
                self.logger.log_train_epoch(f.iter.numpy(), loss_value.numpy(),loss_f.numpy(), loss_bc.numpy(),custom=custom_log_res+custom_log)
                self.losshistory.append(
                    f.iter.numpy(),
                    self.train_state.loss_train,
                    self.train_state.loss_train_bc,
                    self.train_state.loss_train_f,
                    self.train_state.loss_train_x1,
                    self.train_state.loss_train_x2,
                    self.train_state.loss_train_x3,
                    self.train_state.loss_test,
                    None)
                self.varhistory.append(
                    f.iter.numpy(),
                    self.rho.numpy()*rho,
                    self.PI.numpy()*PI)
 
            # if f.iter%200==0:
            #     # #Updating adaptive lambda values
            #     self.lamb_l1,self.lamb_l2,self.lamb_l3,self.lamb_bc=self.GetLambStates(self.lamb_l1,
            #                                                               self.lamb_l2,
            #                                                               self.lamb_l3,
            #                                                               self.lamb_bc,
            #                                                               train_x,
            #                                                               train_y,
            #                                                               uk)
                
            #     # self.lamb_l1,self.lamb_l2,self.lamb_l3,self.lamb_bc=self.GetLambStates(self.lamb_l1,
            #     #                                                                 self.lamb_l2,
            #     #                                                                 self.lamb_l3,
            #     #                                                                 self.lamb_bc,
            #     #                                                                 x_batch_train,y_batch_train,u_batch)
            #     print(f"l1:{self.lamb_l1}, l2:{self.lamb_l2}, l3:{self.lamb_l3}, lbc:{self.lamb_bc}")

            # store loss value so we can retrieve later
            #tf.py_function(f.history.append, inp=[loss_value], Tout=[])

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(self.epoch)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters
        f.history = []

        return f
    @tf.function
    def erro(self,Test): 
        y_pred = self.u_model(Test[0])
        yr=Test[1]
        #erro=tf.sqrt((yr[:,-1,:] - y_pred[:,-1,:])**2)
        erro=tf.square(yr[:,:,:] - y_pred[:,:,0:2])
        return tf.reduce_mean(erro)

    # @tf.function
    # def ED_BCS(self,x,u):
    #     var=[self.rho, self.PI]
    #     #init=time.time()
    #     ddy=dydt(x,tf.constant(1.0,dtype=tf.float32))
    #     #print("Discrete derivative",time.time()-init)
    #     ## Montado o sistema de equacoes
    #     # Tensores (Estados)
    #     #dx=(x[:,1:,:]-x[:,:-1,:])/ts        
    #     # Tensores (Estados atuais preditos)
    #     pbh = x[:,:,0:1]
    #     pwh = x[:,:,1:2]
    #     q = x[:,:,2:] #Vazão
    #     #q=tf.clip_by_value(q,0,100)
    #     #Valores passados de x
    #     # pbhk = xk[:,0:1]
    #     # pwhk = xk[:,1:2]
    #     # qk = xk[:,2:] #Vazão
    #     #Entradas exógenas atuais
    #     ### A entrada da rede exige entradas normalizadas o cálculo dos resíduos não
    #     fq=u[:,:,:1] *60  # desnormalizar para EDO
    #     zc=u[:,:,1:2]*100 # desnormalizar para EDO
    #     pmn=u[:,:,2:]

    #     # Calculo do HEAD e delta de press�o
        
    #     q0 = (q*qc+qmin) / Cq * (f0 / fq)
    #     H0 = -1.2454e6 * q0 ** 2.0 + 7.4959e3 * q0 + 9.5970e2
    #     H = CH * H0 * (fq / f0) ** 2.0  # Head
    #     F1 = 0.158 * ((var[0]*rho* L1 * ((q*qc+qmin)) ** 2.0) / (D1 * A1 ** 2.0)) * (mu / (var[0]*rho* D1 * ((q*qc+qmin)))) ** (1.0/4.0)
    #     F2 = 0.158 * ((var[0]*rho * L2 * ((q*qc+qmin)) ** 2.0) / (D2 * A2 ** 2.0)) * (mu / (var[0]*rho* D2 * ((q*qc+qmin)))) ** (1.0/4.0)
    #     qr = var[1]*PI * (pr - (pbh*pbc+pbmin))
    #     qch = (zc/100.0)*Cc * tf.sqrt(kn)*tf.sqrt(tf.abs(pmn));
    #     ##########################
    #     qch=(qch-qch_lim[0])/qcc
    #     F1=(F1-F1lim[0])/F1c
    #     F2=(F2-F2lim[0])/F2c
    #     H=(H-H_lim[0])/Hc
    #     ###########################

    #     # print('#############')
    #     # print(tf.reduce_mean(tf.square(H)))
    #     # print(tf.reduce_mean(tf.square(F1)))
    #     # print(tf.reduce_mean(tf.square(F2)))
    #     # print(tf.reduce_mean(tf.square(pbh*pbc+pbmin - (pwh*pwc+pwmin) - rho*g*hw)))
    #     # print(tf.reduce_mean(- (1/(qc*M))*(pbh*pbc+pbmin - (pwh*pwc+pwmin) - rho*g*hw - (F1c*F1+F1lim[0]) - (F2c*F2+F2lim[0]) +  rho* g * (H*Hc+H_lim[0]))))
    #     # print(tf.reduce_mean(tf.square(dx[:,:,2:])))
    #     # print('#############')
    #     #return pbh,pwh  # - (1/(qc*M))*(pbh*pbc+pbmin - (pwh*pwc+pwmin) - rho*g*hw - (F1c*F1+F1lim[0]) - (F2c*F2+F2lim[0]) +  rho* g * (H*Hc+H_lim[0]))

    #     # res1=dx[:,:,0:1] - (1/pbc)*b1/V1*(qr - (q*qc+qmin))
    #     # res2=dx[:,:,1:2] - (1/pwc)*b2/V2*((q*qc+qmin) - (qcc*qch+qch_lim[0]))
    #     # res3=dx[:,:,2:] - (1/(qc*M))*(pbh*pbc+pbmin - (pwh*pwc+pwmin) - self.rho*rho*g*hw - (F1c*F1+F1lim[0]) - (F2c*F2+F2lim[0]) +  self.rho*rho* g * (H*Hc+H_lim[0])) 
    #     # return tf.reduce_mean(tf.square(res1)), tf.reduce_mean(tf.square(res2)), tf.reduce_mean(tf.square(res3))

    #     dy1=- (1/pbc)*b1/V1*(qr - (q*qc+qmin))
    #     dy2=- (1/pwc)*b2/V2*((q*qc+qmin) - (qcc*qch+qch_lim[0]))
    #     dy3=- (1/(qc*M))*(pbh*pbc+pbmin - (pwh*pwc+pwmin) - var[0]*rho*g*hw - (F1c*F1+F1lim[0]) - (F2c*F2+F2lim[0]) +  var[0]*rho* g * (H*Hc+H_lim[0]))
    #     return tf.reduce_mean(tf.square(ddy[:,:,0:1]+dy1)), tf.reduce_mean(tf.square(ddy[:,:,1:2]+dy2)), tf.reduce_mean(tf.square(ddy[:,:,2:]+dy3))



    @tf.function    
    def GetLoss(self,y, y_pred,u): 
        #pinn_mode ="off" 0# Turn off all loss terms linked with EDO
        #     "on"1# Turn on the main loss term linked with EDO
        #     "all"2# Turn on all loss terms linked with EDO
        #    "loss2"3# Turn off main loss_EDO and keep new loss EDO
        #remove non measured variable from loss mse error
        ysliced=y_pred[:,:,0:2]
        loss_obs=tf.reduce_mean(tf.square(y - ysliced))
        if self.pinn_mode==1 or self.pinn_mode==2:
            #computing the residues with predicted states
            r1,r2,r3=self.ODE(y_pred,u,self.var)
        else:
            r1,r2,r3=tf.constant(0.0, dtype=tf.float32),tf.constant(0.0, dtype=tf.float32),tf.constant(0.0, dtype=tf.float32)
        
        return loss_obs,r1,r2,r3,(r1+r2+r3)

    #@tf.function
    def GetLamb(self,lamb_bc,X,y,u):
        
        with tf.GradientTape(persistent=True) as tape:
            lb,l1,l2,l3,lf=self.GetLoss(y, self.u_model(X),u)
            
        grad_f = tape.gradient(lf,  self.wrap_training_variables())
        grad_bc = tape.gradient(lb,  self.u_model.trainable_weights)
        del tape
        #print(get_abs_max_grad(grad_f),(1-alfa)*lamb_bc)
        lamb_bc=tf.convert_to_tensor( (1-self.alfa)*lamb_bc+self.alfa*get_abs_max_grad(grad_f)/get_abs_mean_grad(grad_bc),dtype=tf.float32)     
        
        return lamb_bc
    def GetLambStates(self,lamb_l1,lamb_l2,lamb_l3,lamb_bc,X,y,u):
        def update_lamb(l,gradmax,gradx):
            return tf.convert_to_tensor((1-pinn.alfa)*l
                                    +pinn.alfa*get_abs_max_grad(gradmax)/get_abs_mean_grad(gradx),dtype=tf.float32)
        try:
            with tf.GradientTape(persistent=True) as tape:
                lb,l1,l2,l3,lf,R1,R2,R3=self.GetLoss(y, self.u_model(X),u)  
                
            grad_f = tape.gradient(lf,  self.wrap_training_variables())           
            grad_l1 = tape.gradient(l1,  self.wrap_training_variables())
            grad_l2 = tape.gradient(l2,  self.wrap_training_variables())
            grad_l3 = tape.gradient(l3,  self.wrap_training_variables())
            grad_bc = tape.gradient(lb,  self.u_model.trainable_weights) #remember that it does't depend on rho and PI
            del tape
            print("Gradientes")
            print(f'grad_bc:{get_abs_max_grad(grad_bc):1.2e}, grad_1:{get_abs_max_grad(grad_l1):1.2e}, grad_2:{get_abs_max_grad(grad_l2):1.2e}, grad_3:{get_abs_max_grad(grad_l3):1.2e}')
            #print(get_abs_max_grad(grad_f),(1-alfa)*lamb_bc)
            ##### Fixo grad_bc atualiza l1,l2,l3 ############
            # lamb_l1=tf.convert_to_tensor( (1-self.alfa)*lamb_l1
            #                             +self.alfa*get_abs_max_grad(grad_bc)/get_abs_mean_grad(grad_l1),dtype=tf.float32)     
            # lamb_l2=tf.convert_to_tensor( (1-self.alfa)*lamb_l2
            #                             +self.alfa*get_abs_max_grad(grad_bc)/get_abs_mean_grad(grad_l2),dtype=tf.float32)     
            # lamb_l3=tf.convert_to_tensor( (1-self.alfa)*lamb_l3
            #                             +self.alfa*get_abs_max_grad(grad_bc)/get_abs_mean_grad(grad_l3),dtype=tf.float32)     
            
            ##### Fixo l3 atualiza l1,l2,grad_bc ############
                        #print(get_abs_max_grad(grad_f),(1-alfa)*lamb_bc)
            lamb_l1=update_lamb(lamb_l1,grad_l1,grad_l1)
            lamb_l2=update_lamb(lamb_l2,grad_l1,grad_l2)
            lamb_l3=update_lamb(lamb_l3,grad_l1,grad_l3)
            lamb_bc=update_lamb(lamb_bc,grad_l1,grad_bc) 
        except Exception:
            print(traceback.format_exc()) 
        return lamb_l1,lamb_l2,lamb_l3,lamb_bc
    

    def get_params(self):
        rho = self.rho
        PI = self.PI
        return rho

    #@tf.function
    def wrap_training_variables(self):
        var = self.u_model.trainable_weights
        # if self.pinn_mode!=0:
        #     var.extend([self.rho])
        #     var.extend([self.PI])
        #var.extend([self.rho, self.PI])
        
        return  var


    @tf.function
    def GetGradAndLoss(self,y,X,u):
        with tf.GradientTape() as tape:
            #init=time.time()
            #print("Loss computing")
            loss_bc,loss_x1,loss_x2,loss_x3,loss_f = self.GetLoss(y, self.u_model(X),u)   
            #end = time.time()
            #print(f"Runtime computing losses {end - start}")
            loss_f=self.lamb_l1*loss_x1+self.lamb_l2*loss_x2+self.lamb_l3*loss_x3
            
            #loss_value=self.lamb_bc*loss_bc+loss_f
            loss_value=self.lamb_bc*loss_bc+self.lamb_l1*loss_x1+self.lamb_l2*loss_x2+self.lamb_l3*loss_x3
            #loss_value=loss_bc*self.lamb_bc+self.lamb_l1*loss_x1+self.lamb_l2*loss_x2+loss_x3*self.lamb_l3

        grads = tape.gradient(loss_value,  self.wrap_training_variables())
        return grads,loss_bc,self.lamb_l1*loss_x1,self.lamb_l2*loss_x2,self.lamb_l3*loss_x3,loss_value,loss_f
    
    def fit(self, dataset, tf_epochs=5000,adapt_w=False):
        self.logger.set_error_fn(self.erro,dataset.Test)
        train_dataset=dataset.dataset_ADAM
        
        if adapt_w==True:
            self.lamb_l1,self.lamb_l2,self.lamb_l3,self.lamb_bc=self.GetLambStates(self.lamb_l1,
                                                                            self.lamb_l2,
                                                                            self.lamb_l3,
                                                                            self.lamb_bc,
                                                                            train_X,train_y, u_train)
        self.logger.log_train_opt("Adam",self.pinn_mode,self.get_lamb_weights())
        self.logger.start_time=time.time()
        rho = 836.8898;
        PI = 2.7e-8;


        
        
        try:       
            for epoch in range(tf_epochs):
                # Iterate over the batches of the dataset.         
                for step, (x_batch_train, y_batch_train, u_batch) in enumerate(train_dataset):
                    #init=time.time()
                    grads,loss_bc,loss_x1,loss_x2,loss_x3,loss_value,loss_f=self.GetGradAndLoss(y_batch_train,x_batch_train,u_batch)
                    #print(f"Runtime of grad and loss {time.time() - init}")
                    if np.isnan(loss_value.numpy()):
                        print("Nan values appear. Stopping training",loss_x1.numpy(),loss_x2.numpy(),loss_x3.numpy(),loss_bc.numpy())
                        self.logger.log_train_end(tf_epochs,self.train_state)
                        self.summary_train(self.train_state)
                        raise Exception("Loss with Nan values found")
                    #print("save 1")
                    self.train_state.step=str(self.epoch)
                    # print("save 2")
                    self.train_state.rho=self.rho*rho
                    self.train_state.PI=self.PI*PI
                    self.train_state.loss_test=self.erro(dataset.Test).numpy()
                    self.train_state.loss_train=loss_value
                    self.train_state.loss_train_bc=self.lamb_bc*loss_bc.numpy()
                    self.train_state.loss_train_f=loss_f.numpy()
                    self.train_state.loss_train_x1=self.lamb_l1*loss_x1.numpy()
                    self.train_state.loss_train_x2=self.lamb_l2*loss_x2.numpy()
                    self.train_state.loss_train_x3=self.lamb_l3*loss_x3.numpy()
                    self.train_state.weights=self.u_model.get_weights()
                    self.train_state.update_best()
                    
                    self.optimizer.apply_gradients(zip(grads, self.wrap_training_variables()))
                    # end = time.time()
                    # print(f"Runtime of batch  {end - start}")
                if (epoch%300==0 and adapt_w==True):
                    #init=time.time()
                    #self.lamb_l1,self.lamb_l2,self.lamb_l3=self.GetLambStates(self.lamb_l1,self.lamb_l2,self.lamb_l3,x_batch_train,y_batch_train,u_batch)
                    self.lamb_l1,self.lamb_l2,self.lamb_l3,self.lamb_bc=self.GetLambStates(self.lamb_l1,
                                                                                           self.lamb_l2,
                                                                                           self.lamb_l3,
                                                                                           self.lamb_bc,
                                                                                           x_batch_train,y_batch_train,u_batch)
                    print(f'==============Weights===============')
                    print(f'[ bc ,  r1 ,  r2 ,  r3 ]')
                    print(f'{self.get_lamb_weights()}')
                #     self.lamb_bc=self.GetLamb(self.lamb_bc,x_batch_train,y_batch_train,u_batch)   
                #     #print(f"Runtime of update_lamb {time.time() - init}")

                if epoch%20==0:    
                    self.losshistory.append(
                        self.epoch ,
                        self.train_state.loss_train,
                        self.train_state.loss_train_bc,
                        self.train_state.loss_train_f,
                        self.train_state.loss_train_x1,
                        self.train_state.loss_train_x2,
                        self.train_state.loss_train_x3,
                        self.train_state.loss_test,
                        None)
                    self.varhistory.append(
                        self.epoch ,
                        self.rho.numpy()*rho,
                        self.PI.numpy()*PI)     

                custom_log_res=f"|{loss_x1.numpy()*self.lamb_l1.numpy():.2e},{loss_x2.numpy()*self.lamb_l2.numpy():.2e},{loss_x3.numpy()*self.lamb_l3.numpy():.2e}|"
                custom_log=f"{self.rho.numpy()*rho:.1f} {self.PI.numpy()*PI:.2e}"
                self.logger.log_train_epoch(self.epoch, loss_value, loss_f, loss_bc, custom=custom_log_res+custom_log)
                # if self.epoch==100:
                #     #self.optimizer.learning_rate=self.optimizer.learning_rate/2
                #     self.optimizer.learning_rate=0.01
                # if self.epoch==500:
                #         #self.optimizer.learning_rate=self.optimizer.learning_rate/2
                #         self.optimizer.learning_rate=0.001
                self.epoch=self.epoch+1
                self.nsess=self.nsess+tf_epochs # Save epochs for the next session
                self.varhistory.update_best(self.train_state)
            self.logger.log_train_end(tf_epochs,self.train_state)
            self.summary_train(self.train_state)
        
        except Exception as err:
            print(err.args)
            raise      

        return self.losshistory, self.train_state, self.varhistory 


        

    def fit_LBFGS(self, dataset, nt_config):
        #self.logger.log_train_start(self)
        
        self.logger.set_error_fn(self.erro, dataset.Test)
        train_x,train_y, u_train = dataset.dataset_LBFGS

        rho = 836.8898;
        PI = 2.7e-8;
        
        self.logger.log_train_opt("LBFGS",self.pinn_mode,self.get_lamb_weights())
        self.logger.start_time=time.time()
        func=self.function_factory(train_x, train_y,u_train,dataset)
        init_params = tf.dynamic_stitch(func.idx, self.wrap_training_variables())
        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            num_correction_pairs=nt_config.nCorrection,
            tolerance=nt_config.tol,
            parallel_iterations=nt_config.parallel_iter,
            max_iterations=nt_config.maxIter,
            f_relative_tolerance=nt_config.tolFun
            )
        
        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)
        # do some prediction
        #pred_outs = self.u_model(x_batch_train)
        #err = np.abs(pred_outs[:,0:-1]-y_batch_train)
        #print("L2-error norm: {}".format(np.linalg.norm(err)/np.sqrt(11)))
        print("Converged:",results.converged.numpy())
        print("Didn't find a step to satisfy:",results.failed.numpy())
        print("Exausted evaluations:",True if (results.converged.numpy() and results.failed.numpy())==False else False)
        print("Nb evals:",results.num_objective_evaluations.numpy())
        

        self.epoch=func.iter.numpy()
        self.logger.log_train_end(self.epoch,self.train_state)
        self.summary_train(self.train_state)
        return self.losshistory, self.train_state, self.varhistory

    def use_best_weights(self):
        self.u_model.set_weights(self.train_state.best_test_weights)
    def disregard_best_weights(self):
        self.train_state.best_weights=np.inf



    def summary_train(self, train_state):
        print(f"Best model at step:{train_state.best_test_step}, Best rho:{train_state.best_test_rho:.1f}, Best PI:{train_state.best_test_PI:.4e}")
        print("  train loss: {:.2e}".format(train_state.best_test_train))
        print("  test loss: {:.2e}".format(train_state.best_test_loss))
        #print("  test metric: {:s}".format(list_to_str(train_state.best_metrics)))
        # if train_state.best_ystd is not None:
        #     print("  Uncertainty:")
        #     print("    l2: {:g}".format(np.linalg.norm(train_state.best_ystd)))
        #     print(
        #         "    l_infinity: {:g}".format(
        #             np.linalg.norm(train_state.best_ystd, ord=np.inf)
        #         )
        #     )
        #     print(
        #         "    max uncertainty location:",
        #         train_state.X_test[np.argmax(train_state.best_ystd)],
        #     )
    # print("")
        self.is_header_print = False

        #self.logger.log_train_end(tf_epochs + nt_config.maxIter)
    def summary_model(self):
        return self.u_model.summary()
    def plot_predictions(self,train_dataset,test_data_set, q=None):
        predictions = np.array([]).reshape(0,Ns+1)
        yr=np.array([]).reshape(0,Ns)
        yobs=np.array([]).reshape(0,Ns+1)
        yobs_pred=np.array([]).reshape(0,Ns)
        self.u_model.reset_states()
        #Predict over all train data to update states
        for step,(x_batch_train, labels, _,_) in enumerate(train_dataset):
            X_star=tf.reshape(x_batch_train,[x_batch_train.shape[0],1,self.inputs])
            yobs=np.concatenate([yobs,self.u_model(X_star)[:,0:3]],axis=0)
            yobs_pred = np.concatenate([yobs_pred, labels.numpy()[:,0:2]],axis=0)
        # ytr=scaler1.inverse_transform(yobs_pred)
        # yt=scaler1.inverse_transform(yobs)
        ytr=yobs_pred*xc[0:2]+x0[0:2]
        yt=yobs*xc+x0
        for step,(x_batch_test, labels) in enumerate(test_dataset):
            X_star=tf.reshape(x_batch_test,[x_batch_test.shape[0],1,self.inputs])
            predictions = np.concatenate([predictions, self.u_model(X_star)[:,0:3]], axis=0)
            yr = np.concatenate([yr, labels.numpy()[:,0:2]],axis=0)
        #ypred=scaler1.inverse_transform(predictions)
        ypred=predictions*xc+x0
        #y=scaler1.inverse_transform(yr)
        y=yr*xc[0:2]+x0[0:2]
        Fig=plt.figure()
        #ax=Fig.add_subplot()
        t=np.linspace(0,20,ytr.shape[0]+y.shape[0])
        #print(t.shape, yt.shape)
        ax1=Fig.add_subplot(3,1,1)
        ax1.plot(t[:-N_star], yt[:,0],"-b", label="Train observed")
        ax1.plot(t[:-N_star], ytr[:,0],":k")
        ax1.plot(t[-N_star:], ypred[:,0],"-r")
        ax1.plot(t[-N_star:], y[:,0],":k")
        ax1.set_ylabel("Pbh",fontsize=self.Font)
        plt.grid(True)
        #ax1.plot(t[:-N_star], ypred[:,2],":k")
        ax2=Fig.add_subplot(3,1,2)
        ax2.plot(t[:-N_star], yt[:,1],"-b", label="Train observed")
        ax2.plot(t[:-N_star], ytr[:,1],":k")
        ax2.plot(t[-N_star:], y[:,1],":k")
        ax2.plot(t[-N_star:], ypred[:,1],"-r")
        ax2.set_ylabel("Pwh",fontsize=self.Font)
        plt.grid(True)
        ax3=Fig.add_subplot(3,1,3)
        # ax3.plot(t[-N_star:], ypred,":k", label="Test predicted")
        # ax3.plot(t[-N_star:],y, "-b",label="Observed")
        ax3.plot(t[:-N_star], yt[:,2]*3600,"-b", label="Train predicted")
        #ax3.plot(t[:-N_star], ytr[:,2]*3600,":k")
        ax3.plot(t[-N_star:], ypred[:,2]*3600,"-r",label="Test predicted")
        if type(q) is np.ndarray:
            ax3.plot(t, q*3600,":k",label="obs")
        ax3.set_ylabel("q",fontsize=self.Font)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0.9, -0.3), ncol = 3)

        
        return Fig
           
    def predict(self, X):
        y_pred=np.array([]).reshape(0,Ns+1) #one non measured variabel
        yr=np.array([]).reshape(0,Ns)
        for i, (X_star, y_star) in enumerate(X): 
            #X_star = tf.convert_to_tensor(X_star, dtype=self.dtype)
            X_star=tf.reshape(X_star,[X_star.shape[0],1,self.inputs])
            #print(self.u_model(X_star)[:,0:3].shape)
            y_pred=np.concatenate([y_pred, self.u_model(X_star)[:,0:3]],axis=0)
            yr=np.concatenate([yr[:,0:2], y_star.numpy()],axis=0)
        return y_pred, yr