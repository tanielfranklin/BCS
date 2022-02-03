import numpy as np

class Dataset_Pinn(object):
    def __init__(self,dset,normx,normu):
         self.xc,self.x0=normx
         self.uc,self.u0=normu
         self.N=dset[0].shape[0]
         self.data=self._dset_reshape(dset)
         self.xn=self._x_norm()
         self.un=self._u_norm()

    def _dset_reshape(self,dset):
        data=[np.reshape(i,(self.N,1)) for i in dset]
        return np.array(data).T[0]
    
    def _x_norm(self):
        val=[]  
        for i in range(3))]
            val.append(np.array([(value-self.x0[i])/self.xc[i] )
        return np.array(val).T
    def _u_norm(self):
        u=self.data[:4] 
        un=np.array([(i-k)/j for i,j,k in zip(u,self.uc,self.u0)])        
        return un



            
              



