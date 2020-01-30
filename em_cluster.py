#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Siddharth Vadgama
# 1001397508

import os
import sys
import pandas as pd
import numpy as np
from numpy import genfromtxt


# In[2]:


training_file_path=sys.argv[1]
training_file=os.path.basename(training_file_path)
ins = open( training_file, "r" )
data = [[float(n) for n in line.split(',')] for line in ins]
rows,cols=np.shape(data)
dataframe= pd.DataFrame(data)
x=dataframe.drop(columns=[cols-1])
iterations=int(sys.argv[3])
k=int(sys.argv[2])


# In[ ]:





# In[3]:


#initialization
weights = [0 for x in range(k)]
mean = [[0 for x in range(cols-1)] for y in range(k)]
covariance = [[[0 for x in range(cols-1)] for y in range(cols-1)] for z in range(k)]
p_values = [[0 for x in range(k)] for y in range(rows)]
for i in range(np.shape(p_values)[0]):
    ind=np.random.randint(0, k)
    p_values[i][ind]=1
#mean = k*d mean[k]- mean vec for kth guassian
#covariance = k*d*d - co[k] - covar mat for kth guassian
#weights = w[k] - kth guassian weight
p_df=pd.DataFrame(p_values)


# In[4]:


def E(w,m,c,p):
    x1=[]
    for j in range(rows):
        p_x=0.0
        for i in range(k):
            p_x=p_x+P_of_xj(w,m,c,j,i)
        x_temp=[]    
        for i in range(k):
            #print(i)
# Fix this line. PDS turn to 0 not sure why. Producing the right probablilities @
            x3=P_of_xj(w,m,c,j,i)/p_x
            
            x_temp.append(x3)
            #print(x_temp)
        x1.append(x_temp)    
    return pd.DataFrame(x1)


# In[5]:


def P_of_xj(w,m,c,j,i):
    return (Guassian(x.iloc[j],m[i],c[i])*w[i])


# In[6]:


def M(p_df):
    w=w_i(p_df)
    m=u_i(p_df)
    c=sig_i(m,p_df)
    return w,m,c


# In[ ]:


p_df=pd.DataFrame(p_values)
sum(p_df.iloc[:][0].values)
p_sum_k=[]
for i in range(k):
    p_sum_k.append(sum(p_df.iloc[:][i].values))
pt=p_sum_k/sum(p_sum_k)
np.matmul(p_df.values.T,x)[0]


# In[7]:


def w_i(p_df):
    p_sum_k=[]
    for i in range(k):
        p_sum_k.append(sum(p_df.iloc[:][i].values))
    return p_sum_k/sum(p_sum_k)
    


# In[8]:


def u_i(p_df):
    p_sum_k=[]
    mean_num=np.matmul(p_df.values.T,x)
    for i in range(k):
        p_sum_k.append(sum(p_df.iloc[:][i].values))
        mean_num[i]=mean_num[i]/p_sum_k[i]
    return mean_num


# In[ ]:


p_sum_k=[]
mean_num=np.matmul(p_df.values.T,x)
for i in range(k):
    p_sum_k.append(sum(p_df.iloc[:][i].values))
    mean_num[i]=mean_num[i]/p_sum_k[i]
mean_num[0][0]


# In[9]:


def sig_i(mean,p_df):
    p_sum_k=[]
    k_temp=[]
    for i in range(k):
        p_sum_k.append(sum(p_df.iloc[:][i].values))
        r_temp=[]
        for r in range(cols-1):
            c_temp=[]
            for c in range(cols-1):
                #check=np.matmul(p_df.iloc[:][i],x.iloc[:][r].values-mean[i][r]*x.iloc[:][c].values-mean[i][c])/p_sum_k[i]
                check=0
                for j in range(rows):
                    f_part=x.iloc[j][r]-mean[i][r]
                    s_part=x.iloc[j][c]-mean[i][c]
                    check=check+(p_df.iloc[j][i]*f_part*s_part)
                check=check/p_sum_k[i]
                if r==c:
                    if abs(check)<0.0001:
                        check=0.0001
                c_temp.append(check)
            r_temp.append(c_temp)
        k_temp.append(r_temp)
    return k_temp


# In[ ]:


p_df.iloc[0][0]=11


# In[ ]:


p_df


# In[ ]:


mean


# In[ ]:


weights


# In[ ]:


weights,mean,covariance=M(p_df)


# In[10]:


def Guassian(x,u,sig):
    x_u=x-u
    sig_inv=np.linalg.inv(sig)
    in_exp=np.exp(-1*0.5*np.matmul(np.matmul(x_u,sig_inv),x_u.T))
    a=np.power(2*np.pi,cols-1)*np.linalg.det(sig)
    out_exp=1/np.sqrt(a)
    return in_exp*out_exp


# In[ ]:


p_df=E(weights,mean,covariance,p_df)


# In[ ]:


p_df


# In[ ]:


np.power(2*np.pi,cols-1)*np.linalg.det(covariance[0])


# In[ ]:


mean


# In[ ]:


covariance


# In[ ]:


weights


# In[17]:


for o in range(iterations):
    weights,mean,covariance=M(p_df)
    p_df=E(weights,mean,covariance,p_df)
    if o<iterations-1:
        print("After iteration %d" %(o+1))
        for i in range(k):
            print("Weight %d = %.4f   " %(i+1,weights[i]), end=",")
            print("mean %d = (%.4f,%.4f)"%(i+1,mean[i][0],mean[i][1]), end=",")
            print("\n")
print("After final iteration")
for i in range(k):
    print("Weight %d = %.4f  " %(i+1,weights[i]), end=",")
    print("mean %d = (%.4f,%.4f)"%(i+1,mean[i][0],mean[i][1]), end=",")
    print("\n")
    for j in range(np.shape(covariance)[1]):
        print("Sigma %d row %d = (%.4f,%.4f)"%((i+1),(j+1),covariance[i][j][0],covariance[i][j][1]))
    print("\n")

        


# In[ ]:




