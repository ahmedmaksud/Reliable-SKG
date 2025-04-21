def get_msg_from_bitstream(stream,bytesize=8):

    """
    converts numpy array of arbitrary length bits to bytes represented in integer
    discards residual bits
    """
    
    import numpy as np
    from functools import reduce
    
    msg_len=int(np.floor(len(stream)/bytesize))
    msg=np.zeros(msg_len)
    
    for i in range(msg_len):
        temp=stream[bytesize*i:bytesize*(i+1)]
        msg[i]=int(reduce(lambda a,b:2*a+b,temp))
    
    return msg.astype(int)

###############################################################################

def RS_encode(msgA,msgB,chunksize=2,syndsize=1):
    
    """
    Takes messeges of arbitrary length in numpy array interger and gives syndromes
    and messege shunks as 2D array
    """
    
    import numpy as np
    import module_ReedSolo as RS
    RS.init_tables()
    
    msg_chunk=[]
    msg_chunk_B=[]
    msg_synd=[]
    encoded_msg=[]
    i=0
    j=chunksize
    while(j<=len(msgA)):
        temp=msgA[i:j]
        ttemp=msgB[i:j]
        mesecc=RS.rs_encode_msg(temp,syndsize)
        msg_chunk.append(mesecc[0:chunksize])
        msg_chunk_B.append(ttemp)
        msg_synd.append(mesecc[chunksize:])
        encoded_msg.append(mesecc)
        i=j
        j=j+chunksize
    
    msg_chunk=np.array(msg_chunk)
    msg_chunk_B=np.array(msg_chunk_B)
    msg_synd=np.array(msg_synd)
    encoded_msg=np.array(encoded_msg)
    
    return msg_chunk.astype(int),msg_chunk_B.astype(int),msg_synd.astype(int),encoded_msg.astype(int)

###############################################################################

def RS_decode(msg,msg_synd):
    
    """
    takes 2D array of messege and sundrome, puts in new 2D array with indices of decoded
    messeges and number of failed ones
    """
    
    import numpy as np
    import module_ReedSolo as RS
    RS.init_tables()
    
    if len(msg[:,0])!=len(msg_synd[:,0]):
        raise Exception("msg and syndorme not same length")
        
    #chunksize=len(msg[0,:])
    syndsize=len(msg_synd[0,:])
    idx=[]
    corr_msg=np.zeros(np.shape(msg))
    failcount=0

    
    for i in range(len(msg[:,0])):
        temp=np.concatenate((msg[i,:],msg_synd[i,:]))
        try:
            corrected_message, corrected_ecc = RS.rs_correct_msg(temp,syndsize)
            idx.append(i)
            corr_msg[i,:]=corrected_message
            #print(mesecc)
            #print((corrected_message+corrected_ecc))
            #print('\n')
        except RS.ReedSolomonError:
            failcount+=1
            
    if failcount+len(idx)!=len(msg[:,0]):
        raise Exception("decoding fail and success mismatch")
    
    idx=np.array(idx)
    return idx.astype(int),corr_msg,failcount
        
###############################################################################
        
def bits_after_ECC(msgA,msgB,idx,bytesize=8):
    
    """
    extracts bits of apperently corrected messeges and put in 1D numpy array
    """
    
    import numpy as np
    from modules import module_SVD_CEF_advanced as CEFA
    
    strA=[]
    strB=[]
    
    for i in idx:
        tempA=msgA[i,:]
        tempB=msgB[i,:]
        
        strA.append(tempA)
        strB.append(tempB)
        
    strA=np.ravel(np.array(strA))
    strB=np.ravel(np.array(strB))
    
    gray=np.array(CEFA.grayCode(bytesize))
    gray_inda=gray[np.int_(strA)]
    gray_indb=gray[np.int_(strB)]
    
    binary_ind_a=CEFA.get_binary(gray_inda,bytesize)
    binary_ind_b=CEFA.get_binary(gray_indb,bytesize)
    
    binary_ind_a=np.ravel(binary_ind_a)
    binary_ind_b=np.ravel(binary_ind_b)
        
    return binary_ind_a,binary_ind_b
        
###############################################################################

def G_bisectionMethod(target,a=-50,b=50,tol=1e-12):#checked
    
    """
    bisection search for a target in normalized gaussian dist
    """
    
    c=(a+b)/2
    
    while abs(G_CDF_y(c)-target)>tol:
        
        if (G_CDF_y(c)-target)<0 and (G_CDF_y(a)-target)<0:
            a=c
        else:
            b=c
        c=(a+b)/2
    
    return c# scalar float

###############################################################################

def G_train_y_partition(n_parti):#checked
    
    """
    Train equiprobable partition of Nb bits for normalized Gaussianl dist
    """
    
    import numpy as np
    
    partition=np.linspace(0,1,n_parti+1)
    partition=partition[1:-1]
    
    for i in range(len(partition)):
        partition[i]=G_bisectionMethod(target=partition[i])

    partition=np.insert(np.append(partition,[50]),0,-50)
    
    return partition# numpy ndarray of shape (n_parti+1)

###############################################################################

def G_CDF_y(y):#checked
    
    """
    Gives the probability of normalized gaussian dist
    """
    
    from CEF_modules import module_SVD_CEF_advanced as CEFA
    
    return CEFA.normal_to_uniform(y,1)+0.5# numpy ndarray of shape y

###############################################################################

def G_train_y_midpoint(n_mid):
    
    """
    Train equiprobable partition of Nb bits for normalized Gaussianl dist
    """
    
    import numpy as np
    
    partition=np.linspace(0,1,n_mid+1)
    midpoint=0.5*(partition[0:-1]+partition[1:])
     
    for i in range(len(midpoint)):
        midpoint[i]=G_bisectionMethod(target=midpoint[i])
        
    return midpoint# numpy ndarray of shape (n_mid)

###############################################################################

def G_get_indAB_over(SNRx,N,pmmprime,pll,R,partition,midpoint):
    
    """
    Gives the indices of symbols after using overquantization scheme for direct
    qunatization without using CEF
    """

    from functools import reduce
    from CEF_modules import module_SVD_CEF_basic as CEF
    from CEF_modules import module_SVD_CEF_advanced as CEFA
    import numpy as np
    import matplotlib.pyplot as plt
    
    pMprime=2**pmmprime
    
    yy=CEF.gen_x(R)
    yyprime=CEF.add_noise(yy,SNRx)
    
    ind_A=CEFA.get_ind_partition(yy,partition)
    plt.hist(ind_A,2**pll)
    idx_stack=CEFA.get_binary(ind_A,pll)
    
    binary_indA=idx_stack[:,0:pmmprime]
    copoint=idx_stack[:,pmmprime:]
    
    temp_binary=np.arange(pMprime)
    binary=CEFA.get_binary(temp_binary,pmmprime)
    
    ind_B=np.zeros((np.shape(yy)[0],1))
    main_ind_A=np.zeros((np.shape(yy)[0],1))
    for j in range(np.shape(yy)[0]):
        main_ind_A[j]=reduce(lambda a,b:2*a+b,binary_indA[j,:])
           
    for i in range(np.shape(yy)[0]):
        temp=copoint[i:i+1,:]
        temp=np.repeat(temp,pMprime,axis=0)
        temp=np.append(binary,temp,axis=1)
            
        idd=np.zeros((pMprime,1))
            
        for j in range(pMprime):
            idd[j]=reduce(lambda a,b:2*a+b,temp[j,:])
           
        B_points=midpoint[idd.astype(int)]
        ind_B[i]=np.argmin(abs(B_points-yyprime[i]))
        
    return main_ind_A,ind_B

###############################################################################

def get_corrected_bits(N,SNRx,R,pll,pmmprime,chunk_size,ECC_size,partition,midpoint,G_partition,G_midpoint):
    
    """
    Gives apparent corrected bit stream in 1D array with and without CEF after
    using ReedSolo error correction
    """
    
    import numpy as np
    from mofules import module_SVD_CEF_advanced as CEFA
    
    gray=np.array(CEFA.grayCode(pmmprime)) #1D array of gray code indices

    print('CEF section')
    #partition=CEFA.train_y_partition(N,2**pll) #1D array for partition
    #midpoint=CEFA.train_y_midpoint(N,2**pll) #1D array for midpoints

    #directly get indices (2**pmmprime) of AliceBob of the realizations
    ind_A,ind_B=CEFA.get_indAB_over(SNRx,N,pmmprime,pll,R,partition,midpoint)
    print('SER befor gray coding: ',np.count_nonzero(ind_A-ind_B)/np.size(ind_B))

    #convert them to gray indices, work with them from now on, 1D array of int
    gray_inda,gray_indb=gray[np.int_(ind_A[:,0])],gray[np.int_(ind_B[:,0])]
    print('SER after gray coding: ',np.count_nonzero(gray_inda-gray_indb)/np.size(gray_indb))
    
    if np.count_nonzero(ind_A-ind_B)!=np.count_nonzero(gray_inda-gray_indb):
        raise Exception("Gray conversion not correct")
        
    CEF_SER=np.count_nonzero(ind_A-ind_B)/np.size(ind_B)

    #2D array of binary representation for indices
    binary_ind_a,binary_ind_b=CEFA.get_binary(gray_inda,pmmprime),CEFA.get_binary(gray_indb,pmmprime)
    #flatten the array and 1D array bit stream
    binary_ind_a,binary_ind_b=np.ravel(binary_ind_a),np.ravel(binary_ind_b)
    print('BER after gray coding: ',np.count_nonzero(binary_ind_a-binary_ind_b)/np.size(binary_ind_b))
    print('Total gray raw bits: ',len(binary_ind_a))
    
    CEF_BER_b=np.count_nonzero(binary_ind_a-binary_ind_b)/np.size(binary_ind_b)
    CEF_totalb=len(binary_ind_a)

    #breaks the bit stream into byte size and coverts to int, 1D array
    skA,skB=get_msg_from_bitstream(binary_ind_a),get_msg_from_bitstream(binary_ind_b)
    print('Messege Error Rate: ',np.count_nonzero(skA-skB)/np.size(skA))

    msgA,msgB,syndA,encodedA=RS_encode(skA,skB,chunk_size,ECC_size)
    idx,corr_msg,failcount=RS_decode(msgB,syndA)

    bitsA,bitsB=bits_after_ECC(msgA,corr_msg,idx)
    print('BER after RS ECC',np.count_nonzero(bitsA-bitsB)/np.size(bitsA),'chunk size=',chunk_size,'ECC size=',ECC_size)
    print('Total gray corrected bits: ',len(bitsA))
    print('Bit recovery rate: ',len(bitsA)/len(binary_ind_a))
    
    CEF_BER_bprime=np.count_nonzero(bitsA-bitsB)/np.size(bitsA)
    CEF_totalbprime=len(bitsA)
    CEF_eta=len(bitsA)/len(binary_ind_a)

    print('conventional SKG section')
    #G_partition=G_train_y_partition(2**pll) #1D array for partition
    #G_midpoint=G_train_y_midpoint(2**pll) #1D array for midpoints

    #directly get indices (2**pmmprime) of AliceBob of the realizations
    G_ind_A,G_ind_B=G_get_indAB_over(SNRx,N,pmmprime,pll,R,G_partition,G_midpoint)
    print('SER befor gray coding: ',np.count_nonzero(G_ind_A-G_ind_B)/np.size(G_ind_B))

    # convert them to gray indices, work with them from now on, 1D array of int
    G_gray_inda,G_gray_indb=gray[np.int_(G_ind_A[:,0])],gray[np.int_(G_ind_B[:,0])]
    print('SER after gray coding: ',np.count_nonzero(G_gray_inda-G_gray_indb)/np.size(G_gray_indb))

    if np.count_nonzero(G_ind_A-G_ind_B)!=np.count_nonzero(G_gray_inda-G_gray_indb):
        raise Exception("Gray conversion not correct")
        
    DEF_SER=np.count_nonzero(G_gray_inda-G_gray_indb)/np.size(G_gray_indb)

    #2D array of binary representation for indices
    G_binary_ind_a,G_binary_ind_b=CEFA.get_binary(G_gray_inda,pmmprime),CEFA.get_binary(G_gray_indb,pmmprime)
    #flatten the array and 1D array bit stream
    G_binary_ind_a,G_binary_ind_b=np.ravel(G_binary_ind_a),np.ravel(G_binary_ind_b)
    print('BER after gray coding: ',np.count_nonzero(G_binary_ind_a-G_binary_ind_b)/np.size(G_binary_ind_b))
    print('Total gray raw bits: ',len(G_binary_ind_a))
    
    DEF_BER_b=np.count_nonzero(G_binary_ind_a-G_binary_ind_b)/np.size(G_binary_ind_b)
    DEF_totalb=len(G_binary_ind_a)

    #breaks the bit stream into byte size and coverts to int, 1D array
    G_skA,G_skB=get_msg_from_bitstream(G_binary_ind_a),get_msg_from_bitstream(G_binary_ind_b)
    print('Messege Error Rate',np.count_nonzero(G_skA-G_skB)/np.size(G_skA))
    
    G_msgA,G_msgB,G_syndA,G_encodedA=RS_encode(G_skA,G_skB,chunk_size,ECC_size)
    G_idx,G_corr_msg,G_failcount=RS_decode(G_msgB,G_syndA)

    G_bitsA,G_bitsB=bits_after_ECC(G_msgA,G_corr_msg,G_idx) 
    print('BER after RS ECC',np.count_nonzero(G_bitsA-G_bitsB)/np.size(G_bitsA),'chunk size=',chunk_size,'ECC size=',ECC_size)
    print('Total gray corrected bits: ',len(G_bitsA))
    print('Bit recovery rate: ',len(G_bitsA)/len(G_binary_ind_a))
    
    DEF_BER_bprime=np.count_nonzero(G_bitsA-G_bitsB)/np.size(G_bitsA)
    DEF_totalbprime=len(G_bitsA)
    DEF_eta=len(G_bitsA)/len(G_binary_ind_a)
    
    finres=[bitsA,bitsB,G_bitsA,G_bitsB,CEF_SER,CEF_BER_b,CEF_totalb,CEF_BER_bprime,CEF_totalbprime,CEF_eta,DEF_SER,DEF_BER_b,DEF_totalb,DEF_BER_bprime,DEF_totalbprime,DEF_eta]
    
    return finres

###############################################################################

def hex2bin(str):
    
    """
    converts hex string into binary bit string
    """
    
    from tqdm import tqdm
    
    bin = ['0000','0001','0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111']
    aa = ''
    for i in tqdm(range(len(str))):
        aa+=bin[int(str[i],base=16)]
    return aa

###############################################################################

def bitstring_to_bytes(s):
    
    """
    Converts binary bit string into list of byte data structure
    """
    
    v = int(s, 2)
    b = bytearray()
    while v:
        b.append(v & 0xff)
        v >>= 8
    return bytes(b[::-1])

###############################################################################

def get_bytes_AB(bitsA,bitsB,byte_size):
    
    """
    converts bits array to array of multiple bytes
    """
    
    import numpy as np
    
    bytesA=[]
    bytesB=[]
    
    for i in range(int(np.floor(len(bitsA)/(8*byte_size)))):
        chunkA=bitsA[i*8*byte_size:(i+1)*8*byte_size]
        chunkB=bitsB[i*8*byte_size:(i+1)*8*byte_size]
        strA=""
        strB=""
        
        for j in range(len(chunkA)):
            strA+=str(int(chunkA[j]))
            strB+=str(int(chunkB[j]))
            
        bytesA.append(bitstring_to_bytes(strA))
        bytesB.append(bitstring_to_bytes(strB))
        
    return bytesA,bytesB


def get_SHA_bits_from_bytes(G_bytesA,G_bytesB):
    
    """
    Takes list of byte chunks and, runs through SHA 512 and joins all the output
    to obtain binary bitstream
    """
    
    import hashlib
    from tqdm import tqdm
    import numpy as np
    
    digestA=''
    digestB=''
    
    for i in tqdm(range(len(G_bytesA))):
        resultA = hashlib.sha512(G_bytesA[i])
        resultB = hashlib.sha512(G_bytesB[i])
        aaa=resultA.hexdigest()#128 length hex string of 512 bit digest
        bbb=resultB.hexdigest()
    
        digestA+=aaa#join all the hex strings(128*len_G_bytes)
        digestB+=bbb
    
    digbitA=hex2bin(digestA)#convert all the hex to binary string
    digbitB=hex2bin(digestB)

    finbitsA=np.zeros(len(digbitA))
    finbitsB=np.zeros(len(digbitB))

    for i in tqdm(range(len(digbitA))):
        finbitsA[i]=int(digbitA[i])#convert binary string to np array
        finbitsB[i]=int(digbitB[i])
        
    return finbitsA,finbitsB