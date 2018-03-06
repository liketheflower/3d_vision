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

def data_parser(file_name,style =0,scan_lines=99, points_per_scan_line=285):

    #get the cloud point raw data
    cloud_point_data = file_parse(file_name, 10)
    for i in xrange(0,scan_lines*points_per_scan_line,int(points_per_scan_line*(0.1*scan_lines))):
        show_one_scan_line(cloud_point_data,i,points_per_scan_line)
    print cloud_point_data.shape
    clean_ptx = get_clean_ptx(cloud_point_data)
    print clean_ptx.shape
    NUMBER_OF_PLOT_DATA=int(len(clean_ptx))
    tem_file_names = file_name.split('/')
    plot_3d(clean_ptx,tem_file_names[-1][:-4]+'_original.png',True,)
#    x =  clean_ptx[0:NUMBER_OF_PLOT_DATA,0]
 #   y =  clean_ptx[0:NUMBER_OF_PLOT_DATA,1]
  #  z =  clean_ptx[0:NUMBER_OF_PLOT_DATA,2]




if __name__ == "__main__":
    file_names = ["./../DATA/small_example.ptx","./../DATA/big_example.ptx"]
    scan_lines = [99,999]
    points_per_scan_line=[285, 964]
    for i in range(len(file_names)):
        data_parser(file_names[i],i ,scan_lines[i],points_per_scan_line[i])
    





