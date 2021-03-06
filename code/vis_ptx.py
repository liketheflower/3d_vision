import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot_3d(data, file_name='small',save_fig = False, plot_style = 1):
        #{'rear-trunk': 9, 'none': -20, 'misc': 20, 'side-right': 7, 'side-left': 6, 'roof': 1, 'front-bumper': 4, 'trunk': 3, 'interior': 8, 'rear-bumper': 5, 'hood': 2}
    eleven_colors=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00','#ffff33', '#a65628', '#999999','#f781bf','#bababa','#404040','#525252']
    #ax.scatter(data[:,0], data[:,1], data[:,2], ,s=2.5, linewidth=0.1)
    if save_fig:
        ax = plt.axes(projection = '3d')
        if data.shape[0]>0:
        #ax.scatter(data[i,0], data[i,1], data[i,2], c=eleven_colors[color_lbl[i]-1], alpha=0.65)
         #print "color_lbl[i]-1",color_lbl[i]-1
            if plot_style ==0:
                ax.scatter(data[:,0], data[:,1], data[:,2],s=2.5,c=eleven_colors[-1], linewidth=0.1,alpha=0.35)
            if plot_style ==2:
                ax.scatter(data[:,0], data[:,1], data[:,2],s=3.5,c=eleven_colors[-1], linewidth=0.1,alpha=0.99)
            if plot_style ==1:
                ax.scatter(data[:,0], data[:,1], data[:,2],s=1.5,c=eleven_colors[-1], linewidth=0.1,alpha=0.15)
    #ax.scatter(data[:,0], data[:,1], data[:,2], c=lbl, alpha=0.65)
    #ax.set_xlim3d(-80, -40)
    #ax.set_ylim3d(290, 330)
    #ax.set_zlim3d(-36 , -32)
    #plt.show()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(file_name[:-4])
        plt.savefig('./../output/'+file_name, dpi = (200))
        plt.show()
        plt.close()
