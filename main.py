import pandas as pd
from functions import *
import matplotlib.pyplot as plt

dims=np.array([[100,100],[100,500],[50,5000],[50,25000],[50,50000]])
N_dims=dims.shape[0]
outlier_rate=np.array([0.02,0.04,0.06,0.08,0.1])
N_outlier_rate=outlier_rate.shape[0]
N=2
total_time=60*4

# Program

print("ComputationTime :")
times = ComputationTime(dims,outlier_rate,N,total_time)
print("ComputationTime done")

print("HyperparameterSensibility :")
results = HyperparameterSensibility(dims,outlier_rate,N,total_time,times)
print("HyperparameterSensibility done")

print("SimulatedDataEvaluation :")
performance = SimulatedDataEvaluation(dims,outlier_rate,N,total_time,times,results)
print("SimulatedDataEvaluation done")

print("RealDataEvaluation :")
print("data loading :")
data=pd.read_csv("Breast_GSE45827.csv")
data=data.to_numpy()
n_data=data.shape[0]
labels=data[:,1]
data=data[:,2:]
labels[(labels=="luminal_A") | (labels=="luminal_B")]="luminal"
normal_labels=["luminal","basal"]
N_normal_labels=len(normal_labels)
print("data loaded successfully")

performance2 = RealDataEvaluation(data,normal_labels,labels,dims,outlier_rate,N,total_time,times,results)
print("RealDataEvaluation done")

# Vizualisation

    # Dimensionality sensibility

for l in range(4):
    if l==0:
        label_="KNN"
        marker_="+"
    elif l==1:
        label_="MLE"
        marker_="x"
    elif l==2:
        label_="Clustering"
        marker_="o"
    else:
        label_="PCA"
        marker_="^"
    x=dims[:,0]*dims[:,1]
    y=np.mean(times[l],axis=(1,2))
    plt.plot(x,y,label=label_,marker=marker_)
    plt.title("Dimensionality sensibility")
    plt.xlabel("n x p")
    plt.ylabel("computation time (s)")
    plt.legend()
plt.ylim([0,np.max(np.mean(times[[0,2,3]],axis=(2,3)))])
plt.savefig("DimensionalitySensibility.png")
plt.show()

    # Hyperparameter sensibility

for l in range(3):
    for i in range(N_dims):
        n,p=dims[i]
        if l==0:
            label_="KNN"
            x_label_="hyperparameter (K)"
            c_="r"
            n_param=2
        elif l==1:
            label_="MLE"
            x_label_="hyperparameter (threshold)"
            c_="g"
            n_param=1
        else:
            label_="Clustering"
            x_label_="hyperparameter (threshold)"
            c_="b"
            n_param=1
        if results[l,i,0,0,2]==True:
            label_=label_+" + PCA ("+str(n)+" comp)"
        y1,y2=np.zeros(N_outlier_rate*N),np.zeros(N_outlier_rate*N)
        for j in range(N_outlier_rate):
            m=int(outlier_rate[j]*n)
            y1[(j*N):((j+1)*N)]=100*results[l,i,j,:,3].reshape(N)/(n-m)
            y2[(j*N):((j+1)*N)]=100*results[l,i,j,:,4].reshape(N)/m
        for s in range(n_param):
            x=results[l,i,:,:,s].reshape(N*N_outlier_rate)
            plt.plot(x,y1,label="FP rate",marker="+",c="r",linestyle="None")
            plt.plot(x,y2,label="FN rate",marker="x",c="b",linestyle="None")
            if s==1:
                x_label_="hyperparameter (rate)"
            plt.title(label_+", n="+str(n)+", p="+str(p))
            plt.xlabel(x_label_)
            plt.ylabel("error rates (%)")
            plt.legend()
            plt.savefig("HyperParameterSensibility"+str(int(l))+str(int(s))+str(int(i))+".png")
            plt.show()

    # Performance comparison (simulated data) : error rates = f(n*p)

for l in range(3):
    if l==0:
        label_="KNN"
        c_="r"
    elif l==1:
        label_="MLE"
        c_="g"
    else:
        label_="Clustering"
        c_="b"
    x=dims[:,0]*dims[:,1]
    y1=np.zeros(N_dims)
    y2=np.zeros(N_dims)
    for i in range(N_dims):
        n=dims[i,0]
        for j in range(N_outlier_rate):
            m=int(outlier_rate[j]*n)
            y1[i]+=np.mean(performance[l,i,j,:,3])/(n-m)
            y2[i]+=np.mean(performance[l,i,j,:,4])/m
    y1*=100/N_outlier_rate
    y2*=100/N_outlier_rate
    plt.plot(x,y1,label=label_+" FP rate",marker="x",c=c_)
    plt.plot(x,y2,label=label_+" FN rate",marker="+",c=c_)
    plt.title("Performance comparison : error rates = f(n*p)")
    plt.xlabel("n x p")
    plt.ylabel("error rates (%)")
    plt.legend()
plt.savefig("PerformanceComparisonSimulatedDataErrorRates.png")
plt.show()

    # Performance comparison (simulated data) : tables

for i in range(N_dims):
    n,p=dims[i]
    table=np.zeros((4,7)).astype(object)
    table[0]=np.array(["","dimensionality reduction (n° of comp)","hyperparameter(s)","accuracy","FP rate","FN rate","computation time"])
    for l in range(3):
        if l==0:
            table[1+l,0]="KNN"
        elif l==1:
            table[1+l,0]="MLE"
        else:
            table[1+l,0]="Clustering"
        if performance[l,i,0,0,2]==True:
            table[1+l,1]="yes ("+str(n)+")"
        else:
            table[1+l,1]="no"
    table[1,2]="K="+str(int(performance[0,i,0,0,0]))+", rate="+str(performance[l,i,0,0,0])
    for l in range(1,3):
        table[1+l,2]="threshold="+str(performance[l,i,0,0,0])
    table[1:,3]=100*(1-np.mean(performance[:,i,:,:,3]+performance[:,i,:,:,4],axis=(1,2))/n)
    for j in range(N_outlier_rate):
        m=int(outlier_rate[j]*n)
        table[1:,4]+=np.mean(performance[:,i,j,:,3]/(n-m),axis=1)
        table[1:,5]+=np.mean(performance[:,i,j,:,4]/m,axis=1)
    table[1:,4]*=100/N_outlier_rate
    table[1:,5]*=100/N_outlier_rate
    table[1:,6]=np.mean(performance[:,i,:,:,5],axis=(1,2))
    for j in range(4):
        for k in range(7):
            table[j,k]=str(table[j,k])
    np.savetxt("PerformanceComparisonSimulatedDataTables"+str(int(i))+".csv",table,fmt="%s",delimiter=";") 
    print("n="+str(int(n))+", p="+str(int(p))+" :")
    print(table)
    print("")

    # Performance comparison (real data) : tables
    
for i in range(N_normal_labels):
    n=int(np.sum(np.ones(n_data)[labels==normal_labels[i]]))
    table=np.zeros((4,7)).astype(object)
    table[0]=np.array(["","dimensionality reduction (n° of comp)","hyperparameter(s)","accuracy","FP rate","FN rate","computation time"])
    for l in range(3):
        if l==0:
            table[1+l,0]="KNN"
        elif l==1:
            table[1+l,0]="MLE"
        else:
            table[1+l,0]="Clustering"
        if np.any(performance2[l,i,:,:,2])==True:
            table[1+l,1]="yes ("+str(int(n*1.06))+" +- "+str(int(0.04*n))+")"
        else:
            table[1+l,1]="no"
    table[1,2]="K="+str(int(performance2[0,i,0,0,0]))+", rate="+str(performance2[l,i,0,0,1])
    for l in range(1,3):
        table[1+l,2]="threshold="+str(performance2[l,i,0,0,0])
    for j in range(N_outlier_rate):
        m=int(outlier_rate[j]/(1-outlier_rate[j])*n)
        if m==0:
            m=1
        n+=m
        table[1:,4]+=np.mean(performance2[:,i,j,:,3],axis=1)/(n-m)
        table[1:,5]+=np.mean(performance2[:,i,j,:,4],axis=1)/m
        table[1:,3]+=np.mean(performance2[:,i,j,:,3]+performance2[:,i,j,:,4],axis=1)/n
        n-=m
    table[1:,4]*=100/N_outlier_rate
    table[1:,5]*=100/N_outlier_rate
    table[1:,3]*=100/N_outlier_rate
    table[1:,6]=np.mean(performance2[:,i,:,:,5],axis=(1,2))
    for j in range(4):
        for k in range(7):
            table[j,k]=str(table[j,k])
    np.savetxt("PerformanceComparisonRealDataTables"+str(int(i))+".csv",table,fmt="%s",delimiter=";") 
    print("normal_label = "+normal_labels[i]+" ("+str(int(n))+") :")
    print(["dimensionality reduction (n° of comp)","hyperparameter(s)","accuracy","FP rate","FN rate","computation time"])
    print(table)
    print("")
