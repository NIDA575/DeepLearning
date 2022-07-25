import numpy as np
import math
import matplotlib.pyplot as plt
y_8=[]
step = 5/127
for i in range(0, 128):
    x = math.tanh(i*step)
    y_8.append(np.round(x*127))

#print(y)
#plt.plot(y)
#plt.show()

def int8_tanh(x,y):
    for i in range(len(x)):
        if x[i]<0:
            x[i] = -y[-x[i]]
        else:
            x[i] = y[x[i]]
    return x

x = [28, 9, 10, 110, -32767, -10000, -9, -110, -120, 657, -657]
print(x)

y_16 = []
step = 5/32767
print(math.tanh(step*32767)*32767)
exit(0)

for i in range(0, 32768):
    temp = math.tanh(i*step)
    y_16.append(np.round(temp*32767))

print(int8_tanh(x, y_16))
