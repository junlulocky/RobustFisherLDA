import numpy as np
import sys
import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

if len(sys.argv)<2:
	print 'Usage: python plotData1.py <data_file>'

[X, Y] = load.loader(sys.argv[1]).load()
label = map(lambda x:{-1:'red', 1:'blue'}[x], Y)

pca=decomposition.PCA()
pca.n_components=3
X=pca.fit_transform(X)
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],color=label)
plt.show()
exit(0)