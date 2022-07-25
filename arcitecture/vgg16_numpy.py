import numpy as np

def conv(image, kernel):
    image_h = image.shape[0]
    image_w = image.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]


    image_conv = np.zeros([3,3,3])
    print(image.shape)

    #for i in range(h, image_h -h):
    for c in range(0,3):
        for i in range(0, 3):
            for j in range(0, 3):
                sum_1 = 0
                for m in range(kernel_h):
                    k=i
                    for n in range(kernel_w):
                        l=j
                        #sum_1 = sum_1 + kernel[m][n]*image[i-h+m][j-w+n]
                        sum_1 = sum_1 + kernel[c][m][n]*image[c][k][l]
                        l+=1
                    k+=1


                image_conv[c][i][j] = sum_1
    return image_conv

image = np.ones([3,5,5])
kernel = np.ones([6,3,3,3])
output_conv = np.zeros([6,3,3])
#exit(0)
for i in range(6):
    output = conv(image, kernel[i])
    output_conv[i] = output[0,::] + output[1,::] + output[2,::]

print(output_conv)
