import numpy as np
import sys
import random
import load
import math
import util
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

if len(sys.argv)<2:
	print 'Usage: python plotData2.py <data_file> (Adim - Bdim)'

[X, Y] = load.loader(sys.argv[1]).load()
start = 0
dimension = len(X[0])
if len(sys.argv) == 3:
	parts = sys.argv[2].split('-')
	start = int(parts[0])
	dimension = int(parts[1]) - start + 1

row = int(math.floor(math.sqrt(dimension - 1))) + 1
[posX, negX] = util.split(X, Y)
print 'posNum = %d, negNum = %d'%(len(posX), len(negX))
if len(posX) < len(negX):
	gap = len(negX) - len(posX)
	for i in xrange(gap):
		pickup = random.randint(0, len(posX)-1)
		posX.append(posX[pickup])
else:
	gap = len(posX) - len(negX)
	for i in xrange(gap):
		pickup = random.randint(0, len(negX)-1)
		negX.append(negX[pickup])


posX = np.array(posX)
negX = np.array(negX)
fig, axes = plt.subplots(nrows = row, ncols = row)

for i in xrange(dimension):	
	data = DataFrame({'pos':posX[:,i + start], 'neg':negX[:,i + start]},columns = ['pos','neg'])
	data.plot(kind='hist', x= None, y = None, stacked=True, bins=20, legend = False, fontsize = 5, mark_right = False ,ax = axes[i / row, i % row])

plt.show()
