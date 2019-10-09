
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


x_train = np.genfromtxt('X_train.csv', delimiter=',') #return narray
y_train = np.genfromtxt('y_train.csv')
x_test = np.genfromtxt('X_test.csv', delimiter=',')
y_test = np.genfromtxt('y_test.csv')


# ## Part 1

# #### (a)  Forλ= 0,1,2,3,...,5000, solve forwRR.  (Notice that whenλ= 0,wRR=wLS.)  In one figure,plot the 7 values inwRRas a function ofdf(λ). You will need to call a built in SVD function to dothis as discussed in the slides. Be sure to label your 7 curves by their dimension inx.2 
# $w_{rr}=(λI+X^TX)^{−1}X^Ty$

# In[3]:


alphas = np.arange(0, 5001, 1)


# In[4]:


def getRidgeCoef(x, y, lambda_start, lambda_end, n_feature):
    coefs = []
    alphas = np.arange(lambda_start, lambda_end+1,1)
    for num in alphas:
        part1 = np.linalg.inv(num*np.identity(n_feature)+np.matmul(np.transpose(x), x)) #7 features
        part2 = np.matmul(part1, np.transpose(x))
        res = np.matmul(part2, y)
        coefs.append(res)  
    return coefs   


# In[26]:


Wrr = getRidgeCoef(x_train, y_train, 0, 5000, 7)
#Wrr


# In[6]:


#method1: calculate trace
df_x=[]
for num in alphas:
    part1 = np.linalg.inv(np.matmul(np.transpose(x_train), x_train)+num*np.identity(7))
    part2 = np.matmul(x_train, part1)
    part3 = np.matmul(part2, np.transpose(x_train))
    res = np.trace(part3)
    #U,sigma,VT = np.linalg.svd(part3)   #method2: sum diagonal(sigma in SVD)
    df_x.append(res)


# In[7]:


len(df_x)


# In[8]:


Wrr_5000_array = np.asarray(Wrr)
#Wrr_5000_array[:,0] #first column of data

plt.figure(figsize=(10,8))
ax = plt.gca()

l1, = plt.plot(df_x, Wrr_5000_array[:,0], label='cylinders')
l2, = plt.plot(df_x, Wrr_5000_array[:,1], label='displacement')
l3, = plt.plot(df_x, Wrr_5000_array[:,2], label='horsepower')
l4, = plt.plot(df_x, Wrr_5000_array[:,3], label='weight')
l5, = plt.plot(df_x, Wrr_5000_array[:,4], label='acceleration')
l6, = plt.plot(df_x, Wrr_5000_array[:,5], label='year made')
l7, = plt.plot(df_x, Wrr_5000_array[:,6], label='intercept')

plt.legend()
plt.xlabel('df($\lambda$)')
plt.ylabel('coefficients')
plt.title('Ridge coefficients as a function of df($\lambda$)')
plt.show()


# #### (b) Two dimensions clearly stand out over the others. Which ones are they and what information can we get from this?

# 'Weight' and 'Year made' clearly stands out, because their coefficients are significant and different from zero. Year made has positive coefficient, so the greater the year made, the greater the target value; Weight has negative coefficient, so the greater the weight, the smaller the target value.
# 
# The smaller the df($\lambda$), the greater lambda. From the plot, we can see that it's difficult for Ridge regression to shrink coefficients to zero, which is quiet different from LASSO regression.

# #### (c)  For λ= 0,...,50, predict all 42 test cases. Plot the root mean squared error (RMSE) on the testset as a function of λ—not as a function of df(λ). What does this figure tell you when choosing λ for this problem (and when choosing between ridge regression and least squares)?

# In[9]:


#first get w_rr
Wrr_50 = getRidgeCoef(x_train, y_train, 0, 50, 7)
#Wrr_50


# In[10]:


#get prediction result for 42 test cases
Wrr_50_array = np.asarray(Wrr_50)


# In[11]:


pred_test = np.matmul(Wrr_50_array, np.transpose(x_test))
pred_test


# In[12]:


def get_rmse(y, y_pred):
    n = len(y)
    rmse = np.sqrt((1/n) * sum((y_pred - y)**2))
    return rmse


# In[13]:


rmse_list = []
for item in pred_test:
    rmse = get_rmse(y_test,item)
    rmse_list.append(rmse)


# In[14]:


#rmse_list


# In[15]:


alpha2 = np.arange(0, 51, 1)
plt.figure(figsize=(8,6))
ax = plt.gca()

ax.plot(alpha2, rmse_list)
plt.xlabel('$\lambda$')
plt.ylabel('RMSE')
plt.title('RMSE as a function of df($\lambda$)')
plt.show()


# The greater the $\lambda$, the greater the RMSE. RMSE is the smallest when $\lambda$ equals 0. Thus, we tend to choose least square regression instead of ridge regression.

# # Part2

# #### Modify your code to learn apth-order polynomial regression model for p= 1,2,3.  (You’vealready donep= 1above.)  For this implementation use the method discussed in the slides.  Also, be sure to standardize each additional dimension of your data.

# #### d)  In one figure, plot the test RMSE as a function of λ= 0,...,100 for p= 1,2,3.  Based on this plot, which value of p should you choose and why? How does your assessment of the ideal value of λ change for this problem?

# In[16]:


x_train_nosd = np.genfromtxt('X_train.csv', delimiter=',')
x_test_nosd = np.genfromtxt('X_test.csv', delimiter=',')


# In[17]:


#Prepare train dataset
# p = 2
p2_add = np.power(x_train[:, 0:6], 2) #delete column of 1s
p2_data = np.hstack((x_train, p2_add)) #combine polynomial features

# p = 3
p3_add = np.power(x_train[:, 0:6], 3) 
p3_data = np.hstack((p2_data, p3_add))


#Prepare test dataset
# p = 2
p2_add2 = np.power(x_test[:, 0:6], 2) 
p2_test = np.hstack((x_test, p2_add2)) 

# p = 3
p3_add2 = np.power(x_test[:, 0:6], 3) 
p3_test = np.hstack((p2_test, p3_add2))


# In[18]:


len(p2_data[0]) #13 features when p=2


# In[19]:


len(p3_data[0]) #19 featuers when p=3


# Standardize x_test and x_train for p=2, p=3

# In[20]:


for i in range(7, 13):
    p2_test[:,i] = (p2_test[:,i] - np.mean(p2_data[:,i]))/np.std(p2_data[:,i])
    p2_data[:,i] = (p2_data[:,i] - np.mean(p2_data[:,i]))/np.std(p2_data[:,i])

for i in range(7, 19):
    p3_test[:,i] = (p3_test[:,i] - np.mean(p3_data[:,i]))/np.std(p3_data[:,i])
    p3_data[:,i] = (p3_data[:,i] - np.mean(p3_data[:,i]))/np.std(p3_data[:,i])
    #(x_train[:,0] - np.mean(x_train[:,0]))/np.std(x_train[:,0])


# Standardize x_test for p=2, p=3

# In[21]:


#get Wrr for p = 2
Wrr_p1 = getRidgeCoef(x_train, y_train, 0, 100, 7) #when p =1, don't need to add features
Wrr_p2 = getRidgeCoef(p2_data, y_train, 0, 100, 13)
Wrr_p3 = getRidgeCoef(p3_data, y_train, 0, 100, 19)


# In[22]:


#get prediction
pred_test_p1 = np.matmul(Wrr_p1, np.transpose(x_test))
pred_test_p2 = np.matmul(Wrr_p2, np.transpose(p2_test))
pred_test_p3 = np.matmul(Wrr_p3, np.transpose(p3_test)) 
#Wrr_p3: 101(# of lambda)*19(# of featres), p3_test: 42(# of test cases)*19


# In[23]:


#get RMSE list
def get_rmse_list(test, pred):
    rmse_list = []
    for item in pred:
        rmse = get_rmse(test,item)
        rmse_list.append(rmse)
    return rmse_list


# In[24]:


RMSE_p1 = get_rmse_list(y_test, pred_test_p1)
RMSE_p2 = get_rmse_list(y_test, pred_test_p2)
RMSE_p3 = get_rmse_list(y_test, pred_test_p3)


# Plot Graph

# In[25]:


alpha3 = np.arange(0, 101, 1)

plt.figure(figsize=(8,6))
ax = plt.gca()
ax.plot(alpha3, RMSE_p1, c="g", label = 'p=1')
ax.plot(alpha3, RMSE_p2, c="r", label = 'p=2')
ax.plot(alpha3, RMSE_p3, c="b", label = 'p=3')
plt.legend()

plt.xlabel('$\lambda$')
plt.ylabel('RMSE')
plt.title('RMSE as a function of $\lambda$')
plt.show()


# I will choose p = 2.
# 
# Reason: although polynomial regression has smaller RMSE when p = 3, the difference of RMSE between p=2 model and p=3 model is small. Thus, I will choose p = 2 model because it's more simple and has less features. Compared with p=3 model, p=2 model can avoid overfitting problem and increase model generality. 
# 
# When p (degree of polynomial regression) changes, the relation between $\lambda$ and RMSE also changes. Specifically, when p=1, the optimal lambda is 0; when p=2 or p=3, the optimal lambdas are approximately 40. Since RMSE is much greater when p=1 compared with p=2 and p=3, p=1 model (simple Ridge regression) tend to underfit data. 
