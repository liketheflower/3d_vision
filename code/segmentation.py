import numpy as np
import matplotlib.pyplot as plt
import glob
import math

from numpy import linalg as LA

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def get_clean_ptx_2d_array(a):
    # if no point is found, return None, else return the found points in an array with the shape (#POINT,3)
    has_points = True
    clean_ptx =[]
    for i in xrange(a.shape[0]):
        for j in xrange(a.shape[1]):
            if any(a[i,j,k]>0.00001 or a[i,j,k]<-0.0001 for k in range(3)):
                clean_ptx.append(a[i,j])
    if not clean_ptx: 
        has_points = False
        tem_ptx = np.zeros((1, 4))
    else:
        tem_ptx = np.zeros((len(clean_ptx), 4))
        for ii in xrange(len(clean_ptx)):
            tem_ptx[ii] = clean_ptx[ii]
    return  has_points, tem_ptx

def plot_3d(data, file_name='small',save_fig = False, plot_style = 0):
        #{'rear-trunk': 9, 'none': -20, 'misc': 20, 'side-right': 7, 'side-left': 6, 'roof': 1, 'front-bumper': 4, 'trunk': 3, 'interior': 8, 'rear-bumper': 5, 'hood': 2}
    twelve_colors=['#525252','#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00','#ffff33', '#a65628', '#999999','#f781bf','#bababa','#404040']
    #ax.scatter(data[:,0], data[:,1], data[:,2], ,s=2.5, linewidth=0.1)
    if save_fig:
        ax = plt.axes(projection = '3d')
        if data.shape[0]>0:
        #ax.scatter(data[i,0], data[i,1], data[i,2], c=eleven_colors[color_lbl[i]-1], alpha=0.65)
         #print "color_lbl[i]-1",color_lbl[i]-1
            for i in xrange(data.shape[0]):
                if plot_style ==0:
                    ax.scatter(data[i,0], data[i,1], data[i,2],s=2.5,c=twelve_colors[data[i,3]%12], linewidth=0.1,alpha=0.35)
                if plot_style ==1:
                    ax.scatter(data[i,0], data[i,1], data[i,2],s=1.5,c=twelve_colors[data[i,3]%12], linewidth=0.1,alpha=0.15)
    #ax.scatter(data[:,0], data[:,1], data[:,2], c=lbl, alpha=0.65)
    #ax.set_xlim3d(-80, -40)
    #ax.set_ylim3d(290, 330)
    #ax.set_zlim3d(-36 , -32)
    #plt.show()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(file_name)
        plt.savefig('./result/'+file_name, dpi = (200))
        plt.show()
        plt.close()

def get_dot(a,b):
    # get the dot product of a,b
    return np.dot(a,b)


def is_a_point(norms):
    if any(norms[k]>0.0001 or norms[k]<-0.0001 for k in range(3)): return True
    else: return False

def dfs(i,j, norms, ptx, seen, close_norm_thres,label ):
    R, C = norms.shape[0], norms.shape[1]
    open_list = [(i,j)]
    
    # up, down, left, right
    neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    while open_list:
        i, j = open_list.pop()
        ptx[i,j,3] = label
        seen.add((i,j))
        for di,dj in neighbors:
            if 0 <= i+di <R and 0<=j+dj<C and is_a_point(norms[i+di,j+dj]):
                dot_products = get_dot(norms[i,j], norms[i+di,j+dj])
                #print dot_products 
                if abs(dot_products)<=close_norm_thres:
                    open_list.append((i+di,j+dj))
    return ptx, seen

def sequential_lab(ptx, norms,close_norm_thres, abs_file_name ):
    R, C = ptx.shape[0], ptx.shape[1]
    label = 1
    seen = set()
    for i in xrange(R):
        for j in xrange(C):
            print i,j
            if any(norms[i,j,k]>0.0001 or norms[i,j,k]<-0.0001 for k in range(3)) and (i,j) not in seen:
                print "dfs",i,j
                ptx, seen = dfs(i,j, norms, ptx, seen,close_norm_thres,label)
                label += 1
                print label
    data = get_clean_ptx_2d_array(ptx)
    plot_3d(data, abs_file_name,True, 0)
    np.save("./result/"+abs_file_name+"_ptx_with_label_seq.npy", ptx)
    
if __name__ == "__main__":
    close_norm_thres = 0.5
    raw_ptx_files = sorted(glob.glob("*raw_ptx_wo_label.npy"))
    norm_files = sorted(glob.glob("*example_norms.npy"))
    
    for i in xrange(len(raw_ptx_files)):
        ptx,  norms = np.load(raw_ptx_files[i]), np.load(norm_files[i])
        print ptx.shape
        print norms.shape
        ptx[:,:,3] = 0
        abs_file_name =norm_files[i][:-10]
        get_clean_ptx_with_label = sequential_lab(ptx, norms, close_norm_thres,abs_file_name )




