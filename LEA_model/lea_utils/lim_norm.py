import numpy as np
def norm_values():
    #========================================
    # Limites das variáveis para normalização LEA
    #========================================
    flim=[35,65]
    zlim=[5,70]
    pbhlim=(1e5,2.5e5) 
    pwhlim=(2e5,11e5) 
    qlim=(0.1/3600,5/3600)
    def Lim_c(x):
        return x[1]-x[0]
    pbc=Lim_c(pbhlim)
    pwc=Lim_c(pwhlim)
    qc=Lim_c(qlim)
    pbmin=pbhlim[0]
    pwmin=pwhlim[0]
    qmin=qlim[0]
    #xss = np.float32(np.array([8311024.82175957,2990109.06207437,0.00995042241351780]))
    x0=np.array([pbmin,pwmin,qmin]).reshape(1,3)#,0,0])
    xc=np.array([pbc,pwc,qc]).reshape(1,3)#,1,1])
    return xc,x0
