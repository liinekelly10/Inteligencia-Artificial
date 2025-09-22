
import numpy as np

def verifyNeighborhood(image, point, connectivity, counter, ChainCode, SignalLenght):


    if connectivity == 4:

        if image[point[0]-1,point[1]] == 255:
            image[point[0] - 1, point[1]] = 0
            ChainCode.append(0)
            SignalLenght.append(counter)
            counter = counter + 1
            return (point[0]-1,point[1])

        elif image[point[0],point[1]+1] == 255:
            image[point[0], point[1] + 1] = 0
            ChainCode.append(1)
            SignalLenght.append(counter)
            counter = counter + 1
            return  (point[0],point[1]+1)

        elif  image[point[0]+1,point[1]] == 255:
            image[point[0] + 1, point[1]] = 0
            ChainCode.append(2)
            SignalLenght.append(counter)
            counter = counter + 1
            return (point[0]+1,point[1])

        elif  image[point[0],point[1]-1] == 255:
            image[point[0], point[1] - 1] = 0
            ChainCode.append(3)
            SignalLenght.append(counter)
            counter = counter + 1
            return (point[0],point[1]-1)

        else:
            print('none')
    else:
        return point




def normalizeImage(v):
   v = (v - v.min()) / (v.max() - v.min())
   result = (v * 255).astype(np.uint8)
   return result

