import numpy as np
import matplotlib.pyplot as plt
import glob
import math

from numpy import linalg as LA

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



def show_dot(a, resol = 0.2):
    a = abs(a)
    plt.hist(a,bins=10)
    plt.title("histogram of the norms")
    plt.savefig("hist_of_abs_norm.png")
    plt.show()
    discrete_a = np.zeros(a.shape)
    for i in xrange(a.shape[0]):
        for j in xrange(a.shape[1]):
            discrete_a[i,j] =int(a[i,j]/resol)*resol
    plt.close()
    plt.subplot(2,1,1)
    plt.imshow(a)
    plt.title("a) norm map before discrete")
    plt.subplot(2,1,2)
    plt.imshow(discrete_a)
    plt.title("norm map after discrete")
    #plt.show()
    plt.savefig("norm_map.png")
    plt.show()
    plt.close()
    np.save("./result/dot_with_abs_norms_after_discerete.npy", discrete_a)
if __name__ == "__main__":
    dot_norms =np.load("./result/dot_with_abs_norms.npy")
    show_dot(dot_norms)


