def gen_x(N,mode="iid_gaussian",check_unitnorm=False):#checked

    """
    This function generates a Nby1 numpy vector of zero mean unit variance Gaussian RV. The default mode is "iid_gaussian".
    The mode "unitnorm_gaussian" normalizes the norm of the vector.
    """

    import numpy as np
    
    x=np.random.normal(0,1,size=(N,1))
    
    if mode=="unitnorm_gaussian":
        from numpy.linalg import norm
        temp=norm(x,2)
        x=x/temp
        
    if check_unitnorm:
        if abs(norm(x,2)-1)>1e-10:
            raise Exception("x not unit norm")
    
    return x# numpy ndarray of shape (N,1)

###############################################################################

def pdf_y(N,y):#checked
    
    """
    This function calculates the theoretical pdf of given y (the output of CEF) for given N. -1<=y<=1
    """

    import math
    import numpy as np
    
    if np.any(abs(y)>1):
        raise Exception("y value out of bound")
    
    C=(math.gamma(N/2))/((math.sqrt(math.pi))*(math.gamma((N-1)/2)))
    return C*(1-(y**2))**((N-3)/2)# numpy ndarray of shape y

###############################################################################

def cdf_y(N,y):#checked
    
    """
    This function calculates the cdf of given y (the output of CEF) for given N. -1<=y<=1
    """
 
    import numpy as np
    import math
     
    if np.any(abs(y)>1):
        raise Exception("y value out of bound")
    
    C=(math.gamma(N/2))/((math.sqrt(math.pi))*(math.gamma((N-1)/2)))
    thetay=np.arcsin(y)
    return C*cos_integral(N-2,thetay)+0.5# numpy ndarray of shape y
    
###############################################################################

def gen_Q(K,x,eta_thres=2.5,check_Q=False):#checked
    
    """
    This function generates NbyN Q for all l and k. Takes x satisfying eta is below a threshold
    """
    
    import numpy as np
    
    N,_=np.shape(x)
    Q=np.zeros((N,N,N,K))
    
    i=0
    while i<K:
        temp=gen_single_Q(N)
        if get_eta(temp,x)<eta_thres:
            Q[:,:,:,i]=temp
            i=i+1
            
    if check_Q:
        test_Q(Q)
            
    return Q# numpy ndarray of shape (N,N,N,K)

###############################################################################

def gen_single_Q_URP(N,K):#checked
    
    """
    This function generates NbyN Q for all k for URP
    """
 
    import numpy as np
    
    single_Q=np.zeros((N,N,K))
    
    for i in range(K):
        temp=np.random.normal(0,1,(N,N))
        single_Q[:,:,i], _ = np.linalg.qr(temp)
        
    return single_Q# numpy ndarray of shape (N,N,K)

###############################################################################

def gen_single_Q(N):#checked
    
    """
    This function generates NbyN Q for all l and single k
    """
    
    import numpy as np
    
    single_Q=np.zeros((N,N,N))
    
    for i in range(N):
        temp=np.random.normal(0,1,(N,N))
        single_Q[:,:,i], _ = np.linalg.qr(temp)
        
    return single_Q# numpy ndarray of shape (N,N,N)

###############################################################################

def get_eta(single_Q,x):#checked
    
    """
    This function calculates the \eta which is the error performance of CEF
    """
    
    from scipy.linalg import svdvals
    import numpy as np
 
    T=make_T(single_Q,x)
    St=svdvals(T)
    St=St[0:-1]
    
    return np.sqrt((np.inner(St,St))/(len(x)))# numpy float scalar

###############################################################################

def make_T(single_Q,x,check_T=True):#checked
    
    """
    This function calculates T matrix for finding eta
    """
    
    import numpy as np
    from numpy.linalg import eig
    
    N=len(x)
    [_,NQ,_]=np.shape(single_Q)
    
    if check_T:
        if N!=NQ:
            raise Exception("x and Q dimension mismatch")
        
    M=get_M(single_Q,x)
    Z1=M@(M.T)
    
    if check_T:
        Z2=get_MMt_from_xbar(single_Q,x@(x.T))
        from numpy.linalg import norm
        if norm(Z1-Z2,'fro')>1e-10:
            raise Exception("wrong Z")
            
    D,V=eig(Z1)
    idx=np.argsort(-1*D)
    D=D[idx]
    V=V[:,idx]
    
    if any(np.diff(D)>0):
        raise Exception("D not in correct order")
        
    atemp=np.zeros((N,N))
    
    for j in range(1,N):
        atemp1=1/(D[0]-D[j])
        atemp2=V[:,j:j+1]@V[:,j:j+1].T
        atemp3=atemp2*atemp1
        atemp=atemp+atemp3
        
    btemp=np.zeros((N,N))
    
    for l in range(0,N):
        btemp1=x.T@single_Q[:,:,l].T@V[:,0:1]
        btemp2=btemp1*np.eye(N)
        btemp3=x@V[:,0:1].T@single_Q[:,:,l]
        btemp4=btemp2+btemp3
        btemp5=single_Q[:,:,l]@btemp4
        btemp=btemp+btemp5
        
    return atemp@btemp# numpy ndarray of shape (N,N)
        
###############################################################################

def get_M(single_Q,x):#checked
    
    """
    This function calculates M matrix of CEF
    """
    
    import numpy as np
    
    dimM,_,L=np.shape(single_Q)
    M=np.zeros((dimM,L))
    
    for i in range(L):
        M[:,i:i+1]=single_Q[:,:,i]@x
        
    return M# numpy ndarray of shape (N,N)
        
###############################################################################

def get_MMt_from_xbar(single_Q,xbar):#checked
    
    """
    This function calculates MMt from Q and xbar as described in CEF
    """
    
    import numpy as np
    
    M,_,L=np.shape(single_Q)
    ttemp=np.zeros((M,M))
    
    for i in range(L):
        temp1=single_Q[:,:,i]
        temp2=temp1@xbar@temp1.T
        ttemp=ttemp+temp2
        
    return ttemp# numpy ndarray of shape (M,M)

###############################################################################

def test_Q(Q):#checked
    
    """
    This functions cross checks the dimension of all Q and unitarity
    """
    
    import numpy as np
    from numpy.linalg import norm
    
    M,N,L,K=np.shape(Q)
    
    if N!=M or N!=L:
        raise Exception("Q dimension not correct")
        
    for i in range(K):
        for j in range(L):
            if norm((Q[:,:,j,i].T@Q[:,:,j,i])-np.eye(N),'fro')>1e-10:
                raise Exception("Q matrix not unitary")
                
###############################################################################

def get_Y(Q,x,check_Y=False):#checked
    
    """
    This function generates Y matrix according to CEF
    """
    
    import numpy as np
    
    N,_=np.shape(x)
    _,_,_,K=np.shape(Q)
    
    Y=np.zeros((N,K))
    
    for i in range(K):
        Y[:,i:i+1]=get_single_Y(Q[:,:,:,i],x,check_Y)
        
    return Y# numpy ndarray of shape (N,K)

###############################################################################

def get_Y_URP(single_Q,x):#checked
    
    """
    This function generates Y matrix according to URP
    """
    
    import numpy as np
    
    N,_,K=np.shape(single_Q)
    Y=np.zeros((N,K))
    
    for i in range(K):
        Y[:,i:i+1]=single_Q[:,:,i]@x
        
    return Y# numpy ndarray of shape (N,K)
        
###############################################################################

def get_single_Y(single_Q,x,check_Y=False):#checked
    
    """
    This function generates single Y vector according to CEF
    """
    
    from scipy.sparse.linalg import svds
        
    M=get_M(single_Q,x)
    MMt=M@M.T
    
    if check_Y:
        MMt1=get_MMt_from_xbar(single_Q,x@x.T)
        from numpy.linalg import norm
        if norm(MMt-MMt1,'fro')>1e-10:
            raise Exception("MMt not correct")
    
    temp,_,_=svds(MMt,k=1,which='LM')
    
    return temp# numpy ndarray of shape (N,1)
    
###############################################################################

def cos_integral(N,theta):#checked
    
    """
    This function recursively finds the value of integral int(cos^N(x)dx,0,theta)
    """
    
    import numpy as np
    
    if N==0:
        return theta
    elif N==1:
        return np.sin(theta)
    else:
        return ((np.sin(theta)*(np.cos(theta))**(N-1))/(N))+(((N-1)/N)*(cos_integral(N-2,theta)))# numpy ndarray of shape theta
    
###############################################################################

def add_noise(x,SNR):#checked
    
    """
    This function adds white gaussian noise to x according to SNR assuming x is
    iid with unit variance
    """
    
    import numpy as np
    
    w=np.random.normal(0,(1/np.sqrt(SNR)),np.shape(x))
    
    return x+w# numpy ndarray of shape x
    
###############################################################################

def find_SNR(x,xprime):#checked

    """
    This function finds SNR from x and xprime. SNR=var(x)/var(xprime-x)
    """
    import numpy as np
    
    w=xprime-x
    
    return ((np.var(x))/(np.var(w)))# scalar float

###############################################################################

def orient_Y(Y,Yprime):#checked
    
    """
    The function orients all the columns of Yprime according to to original Y
    """
    
    
    import numpy as np
    from numpy.linalg import norm
    
    N1,K1=np.shape(Y)
    N2,K2=np.shape(Yprime)
    
    if N1!=N2 or K1!=K2:
        raise Exception("Y and Yprime dimension mismatch")
        
    newYprime=np.zeros([N1,K1])
    
    for i in range(K1):
        temp=Y[:,i:i+1]
        tempprime=Yprime[:,i:i+1]
        
        if norm(temp+tempprime)<norm(temp-tempprime):
            newYprime[:,i:i+1]=-tempprime
        else:
            newYprime[:,i:i+1]=tempprime
            
    return newYprime# numpy ndarray of shape Yprime

###############################################################################

def get_v_from_x(x):#checked
    
    """
    This function converts iid Gaussian Vector unit norm Vector +1 dim according
    to CEF paper.
    """
    
    import numpy as np
    from numpy.linalg import norm
    
    n1=norm(x)
    n2=norm(np.append(x,[[1]],axis=0))
    
    temp=(1/(n1*n2))*x
    
    return np.append(temp,[[n1/n2]],axis=0)# numpy ndarray of shape (N+1,1)