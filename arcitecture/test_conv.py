import numpy as np

image = np.random.randint(10, size=(3,5, 5))
ker = np.random.randint(10, size=( 3, 3, 3))
out_1 = image[:,:,1] * ker[:,:,1]

