import numpy as np
import flow_vis
import matplotlib.pyplot as plt

def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)

    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    print(width)
    print(height)

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)
    
    
    
flow = readFlow("out.flo")

colored = flow_vis.flow_to_color(flow)

plt.imshow(colored)
plt.axis('off')

plt.show()
