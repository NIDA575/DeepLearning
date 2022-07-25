import numpy as np
import math
import matplotlib.pyplot as plt
step = 6/128

y_8 = []
x_list = []

for i in range(-128,128):
    x = (1/(1+np.exp(-i*step))) 
    x_list.append(i)
    y_8.append(np.round(x*127))

#plt.plot(y_8)
#plt.show()
#print(y_8)

def int8_sigmoid(x, y):
    for i in range(len(x)):
            x[i] = y[x[i]+128]

    return x

x = [ -128,-50, -27, 0, 27, 50, 127]
print(x)
y = int8_sigmoid(x, y_8)
print(y)
plt.xlim([min(x), max(x)])
plt.ylim([min(y), max(y)])
plt.plot(y)
plt.show()
