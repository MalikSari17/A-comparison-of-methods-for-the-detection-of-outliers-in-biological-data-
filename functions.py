import numpy as np
import scipy.stats
import sklearn
import time

# Data sets simulation

def Simulate(n,m,p):
    X=np.zeros((n,p))
    Y=np.zeros(n)
    q=np.zeros(n)
    mean=np.random.uniform(5,15,p)
    sd=np.random.uniform(0.01,4,p)
    X[:(n-m)]=np.random.normal(mean,sd,(n-m,p))
    for i in range(n-m,n):
        q[i]=np.random.randint(0.25*p,0.75*p)
        J=np.random.choice(range(p),int(q[i]),replace=False)
        mean0=np.copy(mean)
        sd0=np.copy(sd)
        for j in J:
            mean0[j]=mean[j]*np.random.uniform(0.5,2)
            sd0[j]=sd[j]*np.random.uniform(0.5,2)
        X[i]=np.random.normal(mean0,sd0,p)
        Y[i]=1
    return X,Y

# Evaluation protocol

def ComputationTime(dims,outlier_rate,N,total_time):
    N_dims=len(dims)
    N_outlier_rate=len(outlier_rate)
    current_time=np.zeros(4)
    times=np.zeros((4,N_dims,N_outlier_rate,N))
    too_long=[False,False,False,False]
    for i in range(N_dims):
        n,p=dims[i]
        for j in range(N_outlier_rate):
            m=int(outlier_rate[j]*n)
            for k in range(N):
                print(100*(i+j/N_outlier_rate+k/(N_outlier_rate*N))/N_dims,"%")
                X,Y=Simulate(n,m,p)
                for l in range(4):
                    if too_long[l]==False:
                        t0=time.time()
                        if l==0:
                            Y_pred=KNN(X,np.random.randint(1,0.25*n),np.random.uniform(0,0.5))
                        elif l==1:    
                            Y_pred=MLE(X,np.random.uniform(0,0.5))
                        elif l==2:    
                            Y_pred=Clustering(X,np.random.uniform(0,0.5))
                        else:
                            X2=PCA(X)
                        t=time.time()-t0
                        current_time[l]+=t
                        times[l,i,j,k]=t
                        if t>(total_time-current_time[l]-t)/((N-k)*(N_dims-i)*(N_outlier_rate-j)):
                            too_long[l]=True
                            t_moy=np.mean(times[l,i][times[l,i]!=0.])
                            times[l,i,(j+1):],times[l,i,j,k:]=t_moy,t_moy
                            times[l,(i+1):]=total_time
    return times

def HyperparameterSensibility(dims,outlier_rate,N,total_time,times):
    N_dims=len(dims)
    N_outlier_rate=len(outlier_rate)
    current_time=np.zeros(3)
    results=np.zeros((3,N_dims,N_outlier_rate,N,5))
    for i in range(N_dims):
        n,p=dims[i]
        too_long=[False,False,False]
        for j in range(3):
            i2=i
            if np.mean(times[j,i2])>(total_time-current_time[j])/(N*N_outlier_rate*(N_dims-i)):
                too_long[j]=True
        for j in range(N_outlier_rate):
            m=int(outlier_rate[j]*n)
            for k in range(N):
                X,Y=Simulate(n,m,p)
                if any(too_long)==True:
                    X2=PCA(X)
                param=np.zeros((3,2))
                print(100*(i+j/N_outlier_rate+k/(N_outlier_rate*N))/N_dims,"%")
                for l in range(3):
                    if too_long[l]==True:
                        X2=PCA(X)
                    if l!=0:
                        param[l,0]=np.random.uniform(0,0.5)
                        param[l,1]=-1
                    else:
                        param[l,0]=np.random.randint(1,0.25*n)
                        param[l,1]=np.random.uniform(0,0.5)
                    t0=time.time()
                    if too_long[l]==True:
                        if l==0:    
                            Y_pred=KNN(X2,param[l,0],param[l,1])
                        elif l==1:    
                            Y_pred=MLE(X2,param[l,0])
                        elif l==2:    
                            Y_pred=Clustering(X2,param[l,0])
                    else:
                        if l==0:    
                            Y_pred=KNN(X,param[l,0],param[l,1])
                        elif l==1:    
                            Y_pred=MLE(X,param[l,0])
                        elif l==2:    
                            Y_pred=Clustering(X,param[l,0])
                    current_time[l]+=time.time()-t0
                    FP=np.sum(Y_pred[Y==0])
                    FN=np.sum(Y[Y_pred==0])
                    results[l,i,j,k]=np.array([param[l,0],param[l,1],too_long[l],FP,FN])
    return results

def SimulatedDataEvaluation(dims,outlier_rate,N,total_time,times,results):
    N_dims=len(dims)
    N_outlier_rate=len(outlier_rate)
    current_time=np.zeros(3)
    performance=np.zeros((3,N_dims,N_outlier_rate,N,6))
    for i in range(N_dims):
        n,p=dims[i]
        param=np.zeros((3,2))
        too_long=[False,False,False]
        for j in range(3):
            i2=i
            if np.mean(times[j,i])>(total_time-current_time[j])/(N*N_outlier_rate*(N_dims-i)):
                too_long[j]=True
                i2=0
            arg_min=np.argmin((results[j,i2,:,:,3]+results[j,i2,:,:,4])/n)
            j_opt,k_opt=arg_min//N,arg_min%N
            param[j,0]=results[j,i2,j_opt,k_opt,0]
            param[j,1]=results[j,i2,j_opt,k_opt,1]
        for j in range(N_outlier_rate):
            m=int(outlier_rate[j]*n)
            for k in range(N):
                X,Y=Simulate(n,m,p)
                if any(too_long)==True:
                    X2=PCA(X)
                print(100*(i+j/N_outlier_rate+k/(N_outlier_rate*N))/N_dims,"%")
                for l in range(3):    
                    t0=time.time()
                    if too_long[l]==True:
                        if l==0:    
                            Y_pred=KNN(X2,param[l,0],param[l,1])
                        elif l==1:    
                            Y_pred=MLE(X2,param[l,0])
                        elif l==2:    
                            Y_pred=Clustering(X2,param[l,0])
                    else:
                        if l==0:    
                            Y_pred=KNN(X,param[l,0],param[l,1])
                        elif l==1:    
                            Y_pred=MLE(X,param[l,0])
                        elif l==2:    
                            Y_pred=Clustering(X,param[l,0])
                    t=time.time()-t0
                    current_time[l]+=t
                    FP=np.sum(Y_pred[Y==0])
                    FN=np.sum(Y[Y_pred==0])
                    performance[l,i,j,k]=np.array([param[l,0],param[l,1],too_long[l],FP,FN,t]) 
    return performance
    
def RealDataEvaluation(data,normal_labels,labels,dims,outlier_rate,N,total_time,times,results):
    n_data,p=data.shape
    N_normal_labels=len(normal_labels)
    N_outlier_rate=len(outlier_rate)
    N_dims=times.shape[1]
    current_time=np.zeros(3)
    performance=np.zeros((3,N_normal_labels,N_outlier_rate,N,6))
    for i in range(N_normal_labels):
        n=int(np.sum(np.ones(n_data)[labels==normal_labels[i]]))
        param=np.zeros((3,2))
        too_long=[False,False,False]
        for l in range(3):
            i2=N_dims-1
            if 3*np.mean(times[l,i2])>(total_time-current_time[l])/(N*N_outlier_rate*(N_normal_labels-i)):
                too_long[l]=True
                i2=0
            arg_min=np.argmin((results[l,i2,:,:,3]+results[l,i2,:,:,4])/dims[i2,0])
            j_opt,k_opt=arg_min//N,arg_min%N
            param[l,0]=results[l,i2,j_opt,k_opt,0]
            param[l,1]=results[l,i2,j_opt,k_opt,1]
        for j in range(N_outlier_rate):
            m=int(outlier_rate[j]/(1-outlier_rate[j])*n)
            if m==0:
                m=1
            n+=m
            for k in range(N):
                outlier_indices=np.random.choice(np.arange(n_data)[labels!=normal_labels[i]],m,replace=False)
                X=np.concatenate((data[labels==normal_labels[i]],data[np.isin(np.arange(n_data),outlier_indices)]))
                Y=np.concatenate((np.zeros(n-m),np.ones(m)))
                print(100*(i+j/N_outlier_rate+k/(N_outlier_rate*N))/N_normal_labels,"%")
                if any(too_long)==True:
                    X2=PCA(X)
                for l in range(3):
                    t0=time.time()
                    if too_long[l]==True:
                        if l==0:    
                            Y_pred=KNN(X2,param[l,0],param[l,1])
                        elif l==1:    
                            Y_pred=MLE(X2,param[l,0])
                        elif l==2:    
                            Y_pred=Clustering(X2,param[l,0])
                    else:
                        if l==0:
                            Y_pred=KNN(X,param[l,0],param[l,1])
                        elif l==1:    
                            Y_pred=MLE(X,param[l,0])
                        elif l==2:    
                            Y_pred=Clustering(X,param[l,0])
                    t=time.time()-t0
                    current_time[l]+=t
                    FP=np.sum(Y_pred[Y==0])
                    FN=np.sum(Y[Y_pred==0])
                    performance[l,i,j,k]=np.array([param[l,0],param[l,1],too_long[l],FP,FN,t])
                    if t>(total_time-current_time[l])/((N-k)*(N_outlier_rate-j)*(N_normal_labels-i)):
                        too_long[l]=True
            n-=m
    return performance

# Methods

def KNN(X,K,rate):
    n,p=X.shape
    Y_pred=np.zeros(n)
    D=np.zeros((n,n))
    d=np.zeros(n)
    for i in range(n-1):
        for j in range(i+1,n):
            D[i,j]=sum((X[i]-X[j])**2)
            D[j,i]=D[i,j]
        d[i]=np.sort(D[i])[int(K)]
    d[n-1]=np.sort(D[i])[int(K)]
    d_threshold=np.mean(np.sort(d)[n-int(rate*n):])
    for i in range(n):
        if d[i]>d_threshold:
            Y_pred[i]=1
    return Y_pred
    
def MLE(X,threshold):
    n,p=X.shape
    Y_pred=np.zeros(n)
    mean=np.mean(X,axis=0)
    sd=np.zeros(p)
    for i in range(n):
        sd=sd+(X[i,:]-mean)**2
    sd=sd/n
    for i in range(n):
        probas=2*np.array([scipy.stats.norm.cdf(mean[j]-np.abs(X[i,j]-mean[j]),mean[j],sd[j]) for j in range(p)])
        proba=np.mean(np.sort(probas)[:int(0.25*p)])
        if proba<threshold:
            Y_pred[i]=1
    return Y_pred

def Clustering(X,threshold):
    n,p=X.shape
    Y_pred=np.zeros(n)
    C_new=np.ones(n)
    C=np.ones((1,n))
    D=np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            D[i,j]=np.sum((X[i]-X[j])**2)
            D[j,i]=D[i,j]
    WCD=np.ones(1)
    s=-2
    s_new=-1.5    
    i_max=0
    while s_new>s:
        C1=np.zeros(n)
        C1_old=np.zeros(n)
        C2=np.zeros(n)
        i1=np.random.choice(np.arange(n)[C_new==1])
        i2=np.random.choice(np.arange(n)[(C_new==1) & (np.arange(n)!=i1)])
        for i in np.arange(n)[C_new==1]:
            if D[i,i1]<D[i,i2]:
                C1[i]=1
            else:
                C2[i]=1
        while any(C1_old!=C1):
            C1_old=np.copy(C1)
            x1=np.mean(X[C1==1])
            x2=np.mean(X[C2==1])
            for i in np.arange(n)[C_new==1]:
                if np.sum((X[i]-x1)**2)<np.sum((X[i]-x2)**2):
                    if C1[i]!=1:
                        C1[i]=1
                        C2[i]=0
                else:
                    if C2[i]!=1:
                        C2[i]=1
                        C1[i]=0
        WCD1=0
        for i in np.arange(n)[C1==1]:
            WCD1+=np.sum(D[i,C1==1])
        WCD1/=np.sum(C1)
        WCD2=0
        for i in np.arange(n)[C2==1]:
            WCD2+=np.sum(D[i,C2==1])
        WCD2/=np.sum(C2)
        WCD=np.append(WCD,WCD1)
        WCD=np.append(WCD,WCD2)
        WCD=np.delete(WCD,i_max,axis=0)
        s=s_new
        s_new=np.mean(WCD)
        if s_new<=s:
            break
        C=np.delete(C,i_max,axis=0)
        C=np.vstack((C,C1,C2))
        for i in range(1,C.shape[0]):
            if WCD[i]>WCD[i_max]:
                i_max=i
        C_new=C[i_max]
    for i in range(C.shape[0]):
        if np.sum(C[i])/n<threshold:
            Y_pred[C[i]==1]=1
    return Y_pred

def PCA(X):
    n=X.shape[0]
    pca=sklearn.decomposition.PCA(n_components=n)
    X2=pca.fit_transform(X)
    return X2





