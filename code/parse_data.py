import numpy as np
import matplotlib.pyplot as plt



from vis_ptx import plot_3d as plot_3d



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
        if sum(ptx_unclean[i,:3]> 0.0001):
            clean_ptx.append(ptx_unclean[i,:3])
    print len(clean_ptx)
    # the last column is reserved for the segmentation label
    clean_ptx_array = np.zeros((len(clean_ptx),4),dtype = np.float)
    for i in xrange(len(clean_ptx)):
        clean_ptx_array[i,:3] = clean_ptx[i]
    return clean_ptx_array
def data_parser(file_name):

    #get the cloud point raw data
    cloud_point_data=file_parse(file_name, 10)
    print cloud_point_data.shape
    clean_ptx = get_clean_ptx(cloud_point_data)
    print clean_ptx.shape
    NUMBER_OF_PLOT_DATA=int(len(clean_ptx))
    tem_file_names = file_name.split('/')
    plot_3d(clean_ptx,tem_file_names[-1][:-4]+'_original.png',True)
#    x =  clean_ptx[0:NUMBER_OF_PLOT_DATA,0]
 #   y =  clean_ptx[0:NUMBER_OF_PLOT_DATA,1]
  #  z =  clean_ptx[0:NUMBER_OF_PLOT_DATA,2]




if __name__ == "__main__":
    file_names = ["./../DATA/small_example.ptx","./../DATA/big_example.ptx"]
    for file_name in file_names:
        data_parser(file_name)
    





