def bisectionMethod(N,target,a=-1,b=1,tol=1e-12):#checked
    
    """
    Bisection search algorithm based on CDF_Y
    """
    
    from CEF_modules import module_SVD_CEF_basic as CEF
    
    c=(a+b)/2
    
    while abs(CEF.cdf_y(N,c)-target)>tol:
        if (CEF.cdf_y(N,c)-target)<0 and (CEF.cdf_y(N,a)-target)<0:
            a=c
        else:
            b=c
        c=(a+b)/2
    
    return c# scalar float

###############################################################################

def T2_inv(w,B=1):#checked
    
    """
    This function transforms a uniform distribution [-1,1] into a triangular distribution [-B,B]
    """

    import numpy as np
    
    if any(abs(w)>1):
        raise Exception("uniform distribution out of bound")
    
    return B*((np.sign(w)*1)-(np.sign(w))*(np.sqrt(1-np.sign(w)*w)))# numpy ndarray of shape w

###############################################################################

def T1(y,N):#checked
    
    """
    This function transforms pdf-y [-1,1] to a uniform RV [-1,1]
    """

    import numpy as np
    import math
    from CEF_modules import module_SVD_CEF_basic as CEF
    
    if any(abs(y)>1):
        raise Exception("y distribution out of bound")
        
    C=(math.gamma(N/2))/((math.sqrt(math.pi))*(math.gamma((N-1)/2)))
    
    return np.sign(y)*2*C*CEF.cos_integral(N-2,np.arcsin(abs(y)))# numpy ndarray of shape y

###############################################################################

def normal_to_uniform(x,B=2):#checked
    
    """
    This function converts unit variance Normal RV to uniform RV [-B/2,B/2]
    """
    
    import numpy as np
    from scipy import special
    
    y = (B/2)*special.erf(x/np.sqrt(2)) # Q(f) = 0.5 - 0.5 erf(f/sqrt(2))
    
    return y# numpy ndarray of shape x

###############################################################################

def train_y_partition(N,n_parti):#checked
    
    """
    This function computes the boundaries of equiprobable quantizer given N and # of partitions n_parti
    """
    
    import numpy as np
    
    partition=np.linspace(0,1,n_parti+1)
    partition=partition[1:-1]
    
    for i in range(len(partition)):
        partition[i]=bisectionMethod(N,target=partition[i])
    
    partition = np.insert(np.append(partition,[1]),0,-1)
        
    return partition# numpy ndarray of shape (n_parti+1)

def PDF_RP(N,x):
    from scipy import special
    import math
    import numpy as np
    a=2**((1-N)/(2))
    b=(abs(x))**((N-1)/(2))
    #c=sp.special.kv(((N-1)/2),(abs(x)))
    c=special.kv(((N-1)/2),(abs(x)))
    d=np.sqrt(np.pi)*math.gamma(N/2)
    return (a*b*c)/d

def train_y_partition_midpoint_RP(N,n_parti):#checked
    
    """
    This function computes the boundaries of equiprobable quantizer given N and # of partitions n_parti
    """
    
    import numpy as np
    
    partition=np.linspace(0,1,n_parti+1)
    midpoint=0.5*(partition[0:-1]+partition[1:])
    partition=partition[1:-1]
    
    xx=np.linspace(-10*np.sqrt(N),10*np.sqrt(N),1000000)
    d=xx[1]-xx[0]
    yy=np.zeros(np.shape(xx))
    yy[0]=PDF_RP(N,xx[0])*d
    for i in range(1,len(xx)):
        yy[i]=yy[i-1]+PDF_RP(N,xx[i])*d
    
    for i in range(len(partition)):
        partition[i]=xx[np.argmin(abs(yy-partition[i]))]
        
    for i in range(len(midpoint)):
        midpoint[i]=xx[np.argmin(abs(yy-midpoint[i]))]
    
    partition = np.insert(np.append(partition,[10*np.sqrt(N)]),0,-10*np.sqrt(N))
        
    return partition,midpoint# numpy ndarray of shape (n_parti+1)

###############################################################################

def train_y_midpoint(N,n_mid):#checked
    
    """
    This function computes the boundaries of equiprobable quantizer given N and # of partitions Nb
    """
    
    import numpy as np
    
    partition=np.linspace(0,1,n_mid+1)
    midpoint=0.5*(partition[0:-1]+partition[1:])
     
    for i in range(len(midpoint)):
        midpoint[i]=bisectionMethod(N,target=midpoint[i])
        
    return midpoint# numpy ndarray of shape (n_mid)

###############################################################################

def get_ind_partition(x,partition):#checked
    
    """
    Returns the indices of partitions where the elements of x falls
    """
    
    import numpy as np
    return np.clip((np.digitize(x,partition)-1),0,len(partition)-2)# numpy ndarray of shape x

###############################################################################

def get_DN(N):#checked
    
    """
    Function to find Dn in CEF_application_V1
    """
    
    import math
    return ((((math.gamma(N/2))**3)*(math.gamma((3*N-7)/(2))))/(math.pi*((math.gamma((N-1)/2))**3)*(math.gamma((3*N-6)/2))))# scalar float

###############################################################################

def get_y_data(N,SNR,R):#checked
    
    """
    Generates sets of y and y^' given the experiment parameters using CEF
    """
    
    import numpy as np
    from CEF_modules import module_SVD_CEF_basic as CEF
    
    K,Ny=N,1
    yy=np.zeros((Ny*K*R,1))
    yyprime=np.zeros((Ny*K*R,1))
    for i in range(R):
        x=CEF.gen_x(N)
        xprime=CEF.add_noise(x,SNR)
        Q=CEF.gen_Q(K,x)
        Y=CEF.get_Y(Q,x)
        Yprime=CEF.get_Y(Q,xprime)
        Yprime=CEF.orient_Y(Y,Yprime)
        
        yy[Ny*K*i:Ny*K*(i+1),0:1]=np.reshape(Y[0:Ny,:],[Ny*K,1])
        yyprime[Ny*K*i:Ny*K*(i+1),0:1]=np.reshape(Yprime[0:Ny,:],[Ny*K,1])
        
    return yy,yyprime# numpy ndarray of shape (Ny*K*R,1)

###############################################################################

def get_y_data_URP(N,SNR,R):#checked
    
    """
    Generates sets of y and y^' given the experiment parameters using URP
    """

    import numpy as np
    from CEF_modules import module_SVD_CEF_basic as CEF
    
    K,Ny=N,1
    yy=np.zeros([Ny*K*R,1])
    yyprime=np.zeros([Ny*K*R,1])
    for i in range(R):
        x=CEF.gen_x(N)
        xprime=CEF.add_noise(x,SNR)
        Q=CEF.gen_single_Q_URP(N,K)
        Y=CEF.get_Y_URP(Q,x)
        Yprime=CEF.get_Y_URP(Q,xprime)
        Yprime=CEF.orient_Y(Y,Yprime)
        
        yy[Ny*K*i:Ny*K*(i+1),0:1]=np.reshape(Y[0:Ny,:],[Ny*K,1]) 
        yyprime[Ny*K*i:Ny*K*(i+1),0:1]=np.reshape(Yprime[0:Ny,:],[Ny*K,1])
        
    return yy,yyprime# numpy ndarray of shape (Ny*K*R,1)

###############################################################################

def get_yy_yyprime_URP(x,xprime,single_Q):#checked
    
    """
    Generates yy and yyprime vectors for URP for Ny=1 which is the conventional output of CEF
    """

    from CEF_modules import module_SVD_CEF_basic as CEF
    import numpy as np
    
    _,_,K=np.shape(single_Q)
    
    Y=CEF.get_Y_URP(single_Q,x)
    Yprime=CEF.get_Y_URP(single_Q,xprime)
    Yprime=CEF.orient_Y(Y,Yprime)
    
    yy=np.reshape(Y[0,:],[K,1])
    yyprime=np.reshape(Yprime[0,:],[K,1])
    
    return yy,yyprime# numpy ndarray of shape (K,1)

###############################################################################

def get_yy_yyprime(x,xprime,Q):#checked
    
    """
    Generates yy and yyprime vectors for Ny=1 which is the conventional output of CEF
    """
    
    from CEF_modules import module_SVD_CEF_basic as CEF
    import numpy as np
    
    _,_,_,K=np.shape(Q)
    
    Y=CEF.get_Y(Q,x)
    Yprime=CEF.get_Y(Q,xprime)
    Yprime=CEF.orient_Y(Y,Yprime)
    
    yy=np.reshape(Y[0,:],[K,1])
    yyprime=np.reshape(Yprime[0,:],[K,1])
    
    return yy,yyprime# numpy ndarray of shape (K,1)

###############################################################################

def get_yy(x,Q):#checked
    
    """
    Generates yy vector only for Ny=1 which is the conventional output of CEF
    """
    
    from CEF_modules import module_SVD_CEF_basic as CEF
    import numpy as np
    
    _,_,_,K=np.shape(Q)
    
    Y=CEF.get_Y(Q,x) 
    yy=np.reshape(Y[0,:],[K,1])

    return yy# numpy ndarray of shape (K,1)

###############################################################################

def get_range(low=1,high=1e8,count=41):#checked
    
    """
    Get logarithmic range from low to high, number of slots is count
    """

    import numpy as np
    temp=np.linspace(np.log10(low),np.log10(high),count)
    return 10**(temp)# numpy ndarray of shape (count)

###############################################################################

def moving_average(x,w):#checked
    
    """
    Moving avg snippet for plotting
    """

    import numpy as np
    temp=np.convolve(x, np.ones(w),'same')/w
    temp[-1]=x[-1]
    temp[0]=x[0]
    
    return temp# numpy ndarray of shape x

###############################################################################

def get_y_dataprev(N,SNRx,R):#checked
    
    """
    Get previously generated data given N,SNR,R
    """
    
    import numpy as np
    import sys
    sys.path.append('../')
    
    yprimedata=np.load('../new_myYprimedata.npy')
    ydata=np.load('../new_myYdata.npy')
    SNRx_set=np.load('../SNR.npy')
    N_set=np.load('../N_set.npy')

    ind_N=np.where(N_set==N)[0][0]
    ind_snr=np.where(SNRx_set==SNRx)[0][0]
    
    yy=ydata[0:R,ind_snr:ind_snr+1,ind_N]
    yyprime=yprimedata[0:R,ind_snr:ind_snr+1,ind_N]
    
    return yy,yyprime# numpy ndarray of shape (R,1)

###############################################################################

def get_binary(yy,pll):#checked
    
    """
    Get pll bit binary of integer vector yy
    """

    import numpy as np
    aa='{0:0'+str(pll)+'b}'
    idx_stack=np.zeros((np.shape(yy)[0],pll)) 
    for i in range(np.shape(yy)[0]):
        idx_stack[i:i+1,:]=np.array(list(aa.format(int(yy[i]))))
    return idx_stack# numpy nd array of shape (len(yy),pll)

###############################################################################

def grayCode(n_bit):#checked
    
    """
    creates a list of gray code indices of n bits
    """
    import numpy as np
    result = [0]
    for i in range(n_bit):
        for n in reversed(result):
            result.append(1 << i | n)
    return np.array(result)# list of size 2**n_bit

###############################################################################

def get_indAB_over(yy,yyprime,SNRx,N,num_of_ind,partition,midpoint):
#    yy,yyprime,SNRx,N,false_constel_size,partition,midpoint
    """
    Gives the indices (int upto num_of_ind) of symbols after using overquantization scheme (len of partition)
    """
    
    from functools import reduce
    import numpy as np
    
    log_num_of_ind=np.log2(num_of_ind).astype(int)
  
    ind_A=get_ind_partition(yy,partition)
    idx_stack=get_binary(ind_A,np.log2(len(midpoint)).astype(int))
    
    binary_indA=idx_stack[:,0:log_num_of_ind]
    copoint=idx_stack[:,log_num_of_ind:]
    
    temp_binary=np.arange(num_of_ind)
    binary=get_binary(temp_binary,log_num_of_ind)
    
    ind_B=np.zeros((np.shape(yy)[0],1))
    main_ind_A=np.zeros((np.shape(yy)[0],1))
    for j in range(np.shape(yy)[0]):
        main_ind_A[j]=reduce(lambda a,b:2*a+b,binary_indA[j,:])
           
    for i in range(np.shape(yy)[0]):
        temp=copoint[i:i+1,:]
        temp=np.repeat(temp,num_of_ind,axis=0)
        temp=np.append(binary,temp,axis=1)
            
        idd=np.zeros((num_of_ind,1))
            
        for j in range(num_of_ind):
            idd[j]=reduce(lambda a,b:2*a+b,temp[j,:])
           
        B_points=midpoint[idd.astype(int)]
        ind_B[i]=np.argmin(abs(B_points-yyprime[i]))
         
    return main_ind_A,ind_B

###############################################################################

def get_ind_indprime_over(num_of_ind,yy,yyprime,partition,midpoint):
    
    from functools import reduce
    import numpy as np
    
    pmmprime=np.log2(num_of_ind).astype(int)
    
    ind_A=get_ind_partition(yy,partition)
    idx_stack=get_binary(ind_A,int(np.log2(len(midpoint))))
    
    binary_indA=idx_stack[:,0:pmmprime]
    copoint=idx_stack[:,pmmprime:]
    
    temp_binary=np.arange(num_of_ind)
    binary=get_binary(temp_binary,pmmprime)
        
    ind_B=np.zeros((np.shape(yy)[0],1))
    main_ind_A=np.zeros((np.shape(yy)[0],1))
    
    for j in range(np.shape(yy)[0]):
        main_ind_A[j]=reduce(lambda a,b:2*a+b,binary_indA[j,:])
           
    for i in range(np.shape(yy)[0]):
        temp=copoint[i:i+1,:]
        temp=np.repeat(temp,num_of_ind,axis=0)
        temp=np.append(binary,temp,axis=1)
            
        idd=np.zeros((num_of_ind,1))
            
        for j in range(num_of_ind):
            idd[j]=reduce(lambda a,b:2*a+b,temp[j,:])
           
        B_points=midpoint[idd.astype(int)]
        ind_B[i]=np.argmin(abs(B_points-yyprime[i]))
         
    return main_ind_A.astype(int),ind_B.astype(int)

###############################################################################

def get_SER(yy,yyprime,N,constel_size,inv_noise_var,mode):
    
    """
    Generates SER given the experiment parameters for CEF and URP as described in ICC paper
    """

    import numpy as np
    from CEF_modules import module_SVD_CEF_basic as CEF
    import matplotlib.pyplot as plt
    
    wid=1
    D=2/constel_size
    sym_set=np.array([(D/2)+D*(i) for i in range(int(constel_size/2))])
    sym_set=np.append(-np.flip(sym_set),sym_set)
    
    if mode=="CEF":
        print("CEF")
        w=T1(yy,N)   
        wprime=T1(yyprime,N)
        plt.hist(w,128)
        plt.show()
    elif mode=="not CEF":
        print("not CEF")
        w=normal_to_uniform(yy,2)   
        wprime=normal_to_uniform(yyprime,2)
        plt.hist(w,128)
        plt.show()
    else:
        raise Exception("No scheme defined")
        
    idx=np.random.randint(0,constel_size,np.shape(yy))
    sym=sym_set[idx]
    my=(((sym+w)+wid)%(2*wid))-wid
    myprime=CEF.add_noise(my,inv_noise_var)
    symprime=(((myprime-wprime)+wid)%(2*wid))-wid
    
    # plt.hist(symprime,512,density=True)
    # plt.xlabel('$s^,_k$')
    # plt.savefig("data_220129/fig_sK_prime_SVD.jpg",dpi=600)
    
    
    idxprime=np.argmin(abs(symprime-sym_set),axis=1)
    idxprime=np.reshape(idxprime,np.shape(yy))
    
    return np.count_nonzero(idx-idxprime)/len(idx)# scalar float

def get_SER_over(yy,yyprime,SNRx,N,constel_size,false_constel_size,inv_noise_var,partition,midpoint):
    
    """
    Gives SER given experiment parameters after using the scheme described in ICC paper
    """
    
    import numpy as np
    from CEF_modules import module_SVD_CEF_basic as CEF
    import matplotlib.pyplot as plt
     
    DM=2/constel_size
    DMprime=2/false_constel_size

    sym_set=np.array([(DM/2)+DM*(i) for i in range(int(constel_size/2))])
    sym_set=np.append(-np.flip(sym_set),sym_set)

    mask_set=np.array([(DMprime/2)+DMprime*(i) for i in range(int(false_constel_size/2))])
    mask_set=np.append(-np.flip(mask_set),mask_set)

    z1,z2=get_indAB_over(yy,yyprime,SNRx,N,false_constel_size,partition,midpoint)

    idx_A=np.random.randint(0,constel_size,np.shape(z1))
    sym_A=sym_set[idx_A]
    mask_A=mask_set[z1.astype(int)]
    
    wid=1
    my=(((sym_A+mask_A)+wid)%(2*wid))-wid
    myprime=CEF.add_noise(my,inv_noise_var)
    
    mask_B=mask_set[z2.astype(int)]
    symprime=(((myprime-mask_B)+wid)%(2*wid))-wid
    plt.hist(symprime,256)
    plt.show()
    
    idxprime=np.argmin(abs(symprime-sym_set),axis=1)
    idxprime=np.reshape(idxprime,np.shape(z1))
    
    return np.count_nonzero(idx_A-idxprime)/len(idx_A)