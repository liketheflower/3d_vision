import numpy as np
import matplotlib.pyplot as plt



from vis_ptx import plot_3d as plot_3d
from numpy import linalg as LA



def file_parse(file_name,head_length):
    line_data=[]
    with open(file_name) as f:
        for line in f:
            line_data.append(line.split())
    #print "before",head_length,line_data[0]
    if head_length >= 1:
        del line_data[0:head_length]
   # print line_data[0]
    line_data_array=np.array(line_data)
    line_data_array = line_data_array.astype(float)
    return line_data_array


def get_clean_ptx(ptx_unclean):
    # this function will remove all the ptx which is (0,0,0)
    clean_ptx = []
    for i in xrange(ptx_unclean.shape[0]):
        if any(ptx_unclean[i,j]> 0.0001 for j in range(3)):
            clean_ptx.append(ptx_unclean[i,:3])
    print len(clean_ptx)
    # the last column is reserved for the segmentation label
    clean_ptx_array = np.zeros((len(clean_ptx),4),dtype = np.float)
    for i in xrange(len(clean_ptx)):
        clean_ptx_array[i,:3] = clean_ptx[i]
    return clean_ptx_array

def show_one_scan_line(ptx, line_no = 100,points_per_scan_line=285):
    one_line = ptx[line_no:line_no+points_per_scan_line*10]
    one_line = get_clean_ptx(one_line)
    #print one_line
    for i in xrange(one_line.shape[0]):
        if any(one_line[i,j]>0 for j in range(3)):
             print one_line[i] 
        if one_line[i,0] < 9 or  one_line[i,2] > -40 :
             #print "before",one_line[i]
             one_line[i]=0
             #print "after",one_line[i]
    plot_3d(one_line,str(line_no/points_per_scan_line)+"_to_"+str(line_no/points_per_scan_line+10)+"_scan_lines.png",True,2)


def get_center(a):
    return np.mean(a,axis = 0)

def a_min_center(a):
    return a-get_center(a)


def get_cov(a):
    """
    cov_ = [xx, xy, xz
            yx, yy, yz
            zx, zy, zz]
    """
    xx,yy,zz,xy,xz,yz = 0, 0 ,0 ,0 ,0 ,0
    for i in xrange(len(a)):
        x,y,z = a[i,0],a[i,1],a[i,2]
        xx += x*x
        yy += y*y
        zz += z*z
        xy += x*y
        xz += x*z
        yz += y*z
    return np.array([[xx, xy, xz], 
                     [xy, yy, yz], 
                     [xz, yz, zz]])


def get_norm(a):
    A = get_cov(a)
    w, v = LA.eig(A)
    return v[:,np.argmin(w)]

def get_clean_ptx_2d_array(a):
    # if no point is found, return None, else return the found points in an array with the shape (#POINT,3)
    clean_ptx =[]
    for i in xrange(a.shape[0]):
        for j in xrange(a.shape[1]):
            if any(a[i,j,k]>0.00001 or a[i,j,k]<-0.0001 for k in range(3)):
                clean_ptx.append(a[i,j])
    if not clean_ptx: return None
    else:
        tem_ptx = np.zeros((len(clean_ptx), 3))
        for ii in xrange(len(clean_ptx)):
            tem_ptx[ii] = clean_ptx[ii]
        return  tem_ptx
def segmentation(ptx, K = 3, TH = 0.4):
    # K is the neighbor width 
    #print a_min_center(ptx)
    R, C = ptx.shape[0], ptx.shape[1]
    ptx_norms = np.zeros((R,C,3))
    for i in xrange(R):
        for j in xrange(C):
            if i-K>=0 and j-K>=0 and i+K<R and j+K <C:
                ptx_after_clean = get_clean_ptx_2d_array(ptx[i-K:i+K+1,j-K:j+K+1,:3])
                if ptx_after_clean== None or ptx_after_clean.shape[0] < (2*K+1)*TH:
                    continue
                else:
                    p_prime = a_min_center(ptx_after_clean)
                    ptx_norms[i,j] = get_norm(p_prime)
                    print ptx_norms[i,j]
  
    #print ptx_norms[:10,:10]
                      

def get_dot(a,b):
    # get the dot product of a,b
    return np.dot(a,b)
def data_parser(file_name,style =0,scan_lines=99, points_per_scan_line=285, plot_ten_scan_lines= False):

    #get the cloud point raw data
    cloud_point_data = file_parse(file_name, 10)
    if plot_ten_scan_lines:
        for i in xrange(0,scan_lines*points_per_scan_line,int(points_per_scan_line*(0.1*scan_lines))):
            show_one_scan_line(cloud_point_data,i,points_per_scan_line)
    print cloud_point_data.shape
    clean_ptx = get_clean_ptx(cloud_point_data)
    print clean_ptx.shape
    NUMBER_OF_PLOT_DATA=int(len(clean_ptx))
    tem_file_names = file_name.split('/')
    plot_3d(clean_ptx,tem_file_names[-1][:-4]+'_original.png',False,style)
    return cloud_point_data
#    x =  clean_ptx[0:NUMBER_OF_PLOT_DATA,0]
 #   y =  clean_ptx[0:NUMBER_OF_PLOT_DATA,1]
  #  z =  clean_ptx[0:NUMBER_OF_PLOT_DATA,2]




if __name__ == "__main__":
    file_names = ["./../DATA/small_example.ptx","./../DATA/big_example.ptx"]
    scan_lines = [99,999]
    points_per_scan_line=[285, 964]
    """
    ptx = np.array([[0, 0, 0], 
                    [1, 1, 1], 
                    [2, 2, 2]])
    segmentation(ptx)
    """
    for i in range(len(file_names)-1):
        R, C = scan_lines[i],points_per_scan_line[i]
        ptx = data_parser(file_names[i],i ,R, C)
        ptx_reshape = ptx.reshape(R, C, ptx.shape[-1])
        segmentation(ptx)
    





