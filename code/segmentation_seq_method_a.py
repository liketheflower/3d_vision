# this code is sequential labelling by using the method a

import numpy as np
import matplotlib.pyplot as plt
import glob
import math

from numpy import linalg as LA

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import collections
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

def plot_3d(data, file_name='small',save_fig = False, plot_style = 0,color_label_dict={}):
        #{'rear-trunk': 9, 'none': -20, 'misc': 20, 'side-right': 7, 'side-left': 6, 'roof': 1, 'front-bumper': 4, 'trunk': 3, 'interior': 8, 'rear-bumper': 5, 'hood': 2}
    twelve_colors=['#525252','#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00','#ffff33', '#a65628', '#999999','#f781bf','#bababa','#404040']
    five_colors = ['#ff0000','#00ff00','#0000ff','#d95f02','#404040'] 
    #ax.scatter(data[:,0], data[:,1], data[:,2], ,s=2.5, linewidth=0.1)
    
    if save_fig:
        ax = plt.axes(projection = '3d')
        ax.grid(False)
        
        if data.shape[0]>0:
        #ax.scatter(data[i,0], data[i,1], data[i,2], c=eleven_colors[color_lbl[i]-1], alpha=0.65)
         #print "color_lbl[i]-1",color_lbl[i]-1
            for i in xrange(data.shape[0]):
                if plot_style ==0:
                   # print data[i,3], type(data[i,3])
                    col_idx = color_label_dict[int(data[i,3])]
                    #print col_idx, type(col_idx)
                    ax.scatter(data[i,0], data[i,1], data[i,2],s=3.5,c=five_colors[col_idx], linewidth=0.1,alpha=0.35)
                if plot_style ==1:
                    ax.scatter(data[i,0], data[i,1], data[i,2],s=1.5,c=twelve_colors[int(data[i,3])%12], linewidth=0.1,alpha=0.15)
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

def dfs(i,j, norms, ptx, close_norm_thres,label ,label_points_count):
    R, C = norms.shape[0], norms.shape[1]
    open_list = [(i,j)]
    
    # up, down, left, right
    base_neighbor = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1)]
    M = 4
    neighbors =[]
    for i in xrange(1,M):
       tem = [(i*a,i*b) for a,b in base_neighbor ] 
       neighbors.extend(tem)  
    while open_list:
        i, j = open_list.pop()
        ptx[i,j,3] = label
        label_points_count[label]+=1
  #      seen.add((i,j))
        for di,dj in neighbors:
            if 0 <= i+di <R and 0<=j+dj<C and any(abs(ptx[i+di,j+dj,k])>0.001 for k in range(3)) and ptx[i+di,j+dj,3]<0.5:
                #print dot_products 
                if abs(get_dot(norms[i,j], norms[i+di,j+dj])) > close_norm_thres:
                    open_list.append((i+di,j+dj))
    return ptx,label_points_count

def sequential_lab_a(ptx, norms,close_norm_thres, abs_file_name ):
    R, C = ptx.shape[0], ptx.shape[1]
    label = 1
    label_points_count = collections.defaultdict(int)
    #seen = set()
    for i in xrange(R):
        for j in xrange(C):
            print i,j
            if any(abs(ptx[i,j,k])>0.001 for k in range(3)) and ptx[i,j,3]<0.5:
                print "dfs",i,j
                ptx,label_points_count = dfs(i,j, norms, ptx ,close_norm_thres,label,label_points_count)
                label += 1
                print label
    print label_points_count
    color_label_dict = {}
    label_count = [(k,v) for k,v in label_points_count.items()]
    label_count.sort(key=lambda x: x[1],reverse = True)
    print label_count
    top_labels =collections.defaultdict(int)
    # show top 4 planes and set the rest with color label 4
    for i in xrange(len(label_count)):
        if i<4:
            top_labels[i]+=label_count[i][1]
            color_label_dict[label_count[i][0]] = i
        else:
            top_labels[4]+=label_count[i][1]
            color_label_dict[label_count[i][0]] = 4
    color_label_dict[0] = 4
    print color_label_dict
    print "top labels", top_labels
    plt.figure()
    plt.scatter(label_points_count.keys(),label_points_count.values())
    plt.xlabel("label id")
    plt.ylabel("# cloud points")
    plt.title("# could points of different labels of sequential labelling method a")
    plt.show()
    plt.close()
    _,data = get_clean_ptx_2d_array(ptx)
    print "type data",type(data)
    plot_3d(data, abs_file_name+"_seq_a",True, 0,color_label_dict)
    np.save("./result/"+abs_file_name+"_ptx_with_label_seq_a.npy", ptx)

if __name__ == "__main__":
    close_norm_thres = 0.99
    raw_ptx_files = sorted(glob.glob("*raw_ptx_wo_label.npy"))
    norm_files = sorted(glob.glob("*example_norms.npy"))

    for i in xrange(len(raw_ptx_files)):
        ptx,  norms = np.load(raw_ptx_files[i]), np.load(norm_files[i])
        ptx[:,:,3] = 0
 #       dot_norms =np.load("./result/dot_with_abs_norms_after_discerete.npy") 
        sequential_lab_a(ptx,norms,close_norm_thres,"small_example")


