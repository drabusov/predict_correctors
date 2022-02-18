#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle

from correction import *


# In[2]:


def get_errors(filename):
    dk_arr = pickle.load(open(filename, "rb"))
    
    madx = Madx(stdout=False)
    
    Qx, Qy = 18.75, 18.8 # weak point, the code can break here
    twiss, twiss_cold_arr = lattice_setup(Qx,Qy, madx)
    
    return add_quadrupole_errors(twiss, dk_arr, madx, 0, store_errors=False)


# In[3]:


def transform_df(df, twiss):
    
    bpmx = twiss[twiss["keyword"].str.contains("hmonitor")][["name", "betx"]]
    bpmy = twiss[twiss["keyword"].str.contains("vmonitor")][["name", "bety"]]

    for name, beta in zip(bpmx["name"], bpmx["betx"]):
        df[name[:-2]] = [beta]

    for name, beta in zip(bpmy["name"], bpmy["bety"]):
        df[name[:-2]] = [beta]
        
    return df


# In[4]:


import os
path_out, path_err = './results', './results' #'./restore_errors'
err_names = [f"{path_err}/{name}" for name in os.listdir(path_err) 
             if os.path.isfile(f"{path_err}/{name}") and "error" in name]
out_names = [f"{path_out}/{name}" for name in os.listdir(path_out) 
             if os.path.isfile(f"{path_out}/{name}") and "out" in name]


# In[6]:


out = []
for err_name in err_names:
    twiss_err = get_errors(err_name)
    idx = err_name.strip(".p").split("_")[-1]
    
    out_name = f"{path_out}/output_{idx}.csv"
    if out_name in out_names:
        df0 = pd.read_csv(out_name)
        tmp = transform_df(df0, twiss_err)
        out.append(tmp)
    else:
        print(f"trouble occured with filenames: {err_name} and {out_name}")


# In[7]:


df = pd.concat(out)


# In[8]:


df.to_csv("./results/betabeat_and_correctors.csv", index=False)






