#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# show the bias in sample variance

# vars for IQ
pop_mean = 100.0
pop_sigma = 15.0
pop_size = 1000000

population = np.random.normal(pop_mean, pop_sigma, pop_size)

_ = plt.hist(population, bins='auto')
plt.show()


# In[6]:


# sample size of 30 from population
population[:10]
num_samps = 10000

#n=30
samps_mini = np.random.choice(population, (num_samps, 30), replace=True)
#n=100
samps_low = np.random.choice(population, (num_samps, 100), replace=True)
#n=1000
samps_mid = np.random.choice(population, (num_samps, 1000), replace=True)
#n=10000
samps_hi = np.random.choice(population, (num_samps, 10000), replace=True)

# example sample to show non-normal
_ = plt.hist(samps_mini[3], bins=20)
plt.show()
samps_mini.shape


# In[20]:


means_mini = np.mean(samps_mini, axis=1)
means_low = np.mean(samps_low, axis=1)
means_mid = np.mean(samps_mid, axis=1)
means_hi = np.mean(samps_hi, axis=1)

n_bins = 20
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(40,25))
ax0, ax1, ax2, ax3 = axes.flatten()

ax0.hist(means_mini, n_bins, density=True, histtype='bar')
ax0.legend(prop={'size':10})
ax0.set_title('mu for n=30: {}, std error:{}'.format(round(np.mean(means_mini),2), round(np.std(samps_mini)),2), fontsize=30)

ax1.hist(means_low, n_bins, density=True, histtype='bar')
ax1.legend(prop={'size':10})
ax1.set_title('mu for n=100: {}, std error:{}'.format(round(np.mean(means_low),2), round(np.std(samps_low)),2), fontsize=30)

ax2.hist(means_mid, n_bins, density=True, histtype='bar')
ax2.legend(prop={'size':10})
ax2.set_title('mu for n=1000: {}, std error:{}'.format(round(np.mean(means_mid),2), round(np.std(samps_mid)),2), fontsize=30)

ax3.hist(means_hi, n_bins, density=True, histtype='bar')
ax3.legend(prop={'size':10})
ax3.set_title('mu for n=10k: {}, std error:{}'.format(round(np.mean(means_hi),2), round(np.std(samps_hi)),2), fontsize=30)


# In[21]:


# calculate var with n, n-1, n+1, compare with population variance
pop_var = np.var(population)
pop_var

# for n=30
var_mini_n = np.zeros(num_samps, dtype=np.float64)
np.var(samps_mini, axis=1, dtype=np.float64, out=var_mini_n)
var_mini_dof = np.zeros(num_samps, dtype=np.float64)
np.var(samps_mini, axis=1, dtype=np.float64, out=var_mini_dof, ddof=1)

# for n=100
var_low_n = np.zeros(num_samps, dtype=np.float64)
np.var(samps_low, axis=1, dtype=np.float64, out=var_low_n)
var_low_dof = np.zeros(num_samps, dtype=np.float64)
np.var(samps_low, axis=1, dtype=np.float64, out=var_low_dof, ddof=1)

# for n=1000
var_mid_n = np.zeros(num_samps, dtype=np.float64)
np.var(samps_mid, axis=1, dtype=np.float64, out=var_mid_n)
var_mid_dof = np.zeros(num_samps, dtype=np.float64)
np.var(samps_mid, axis=1, dtype=np.float64, out=var_mid_dof, ddof=1)

# for n=10k
var_hi_n = np.zeros(num_samps, dtype=np.float64)
np.var(samps_hi, axis=1, dtype=np.float64, out=var_hi_n)
var_hi_dof = np.zeros(num_samps, dtype=np.float64)
np.var(samps_hi, axis=1, dtype=np.float64, out=var_hi_dof, ddof=1)

n_bins = 100
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(80,70))
ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes.flatten()

# style
#plt.rcParams['axes.titlesize'] = 20

ax0.hist(var_mini_n, n_bins, density=True, histtype='bar')
#ax0.title.set_text('test 1')
ax0.set_title('mu var for n=30, bias: {}'.format(round(np.mean(var_mini_n)),2), fontsize=60)


ax1.hist(var_mini_dof, n_bins, density=True, histtype='bar')
ax1.set_title('mu var for n=30, unbias: {}'.format(round(np.mean(var_mini_dof)),2), fontsize=60)

ax2.hist(var_low_n, n_bins, density=True, histtype='bar')
ax2.set_title('mu var for n=100, bias: {}'.format(round(np.mean(var_low_n)),2), fontsize=60)
              
ax3.hist(var_low_dof, n_bins, density=True, histtype='bar')
ax3.set_title('mu var for n=100, unbias: {}'.format(round(np.mean(var_low_dof)),2), fontsize=60)
              
ax4.hist(var_mid_n, n_bins, density=True, histtype='bar')
ax4.set_title('mu var for n=1k, bias: {}'.format(round(np.mean(var_mid_n)),2),fontsize=60)
              
ax5.hist(var_mid_dof, n_bins, density=True, histtype='bar')
ax5.set_title('mu var for n=1k, unbias: {}'.format(round(np.mean(var_mid_dof)),2),fontsize=60)
              
ax6.hist(var_hi_n, n_bins, density=True, histtype='bar')
ax6.set_title('mu var for n=10k, bias: {}'.format(round(np.mean(var_hi_n)),2),fontsize=60)
              
ax7.hist(var_hi_dof, n_bins, density=True, histtype='bar')
ax7.set_title('mu var for n=10k, unbias: {}'.format(round(np.mean(var_hi_dof)),2),fontsize=60)


# In[ ]:


#bootstrap from sample of size n, a sample of size n

