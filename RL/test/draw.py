import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

path = [[2,2],[2,3],[2,4]]
x,y = zip(*path)
plt.plot(x,y)
plt.show()
