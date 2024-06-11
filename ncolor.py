"""
Implementation of ncolor theorm in python from: https://forum.image.sc/t/relabel-with-4-colors-like-map/33564/6
"""

import numpy as np
from numba import jit
import random
from scipy.ndimage import generate_binary_structure

def neighbors(shape, conn=1):
    dim = len(shape)
    block = generate_binary_structure(dim, conn)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

@jit(nopython=True)
def search(img, nbs):
    s, line = 0, img.ravel()
    rst = np.zeros((len(line),2), img.dtype)
    for i in range(len(line)):
        if line[i]==0:continue
        for d in nbs:
            if line[i+d]==0: continue
            if line[i]==line[i+d]: continue
            rst[s,0] = line[i]
            rst[s,1] = line[i+d]
            s += 1
    return rst[:s]
                            
def connect(img, conn=1):
    buf = np.pad(img, 1, 'constant')
    nbs = neighbors(buf.shape, conn)
    rst = search(buf, nbs)
    if len(rst)<2: return rst
    rst.sort(axis=1)
    key = (rst[:,0]<<16)
    key += rst[:,1]
    order = np.argsort(key)
    key[:] = key[order]
    diff = key[:-1]!=key[1:]
    idx = np.where(diff)[0]+1
    idx = np.hstack(([0], idx))
    return rst[order][idx]

def mapidx(idx):
    dic = {}
    for i in np.unique(idx): dic[i] = []
    for i,j in idx:
        dic[i].append(j)
        dic[j].append(i)
    return dic

# give a touch map: {1:[2,3], 2:[1], 3:[1]}
def render_net(conmap, n=4, rand=12, shuffle=True):
    nodes = list(conmap.keys())
    colors = dict(zip(nodes, [0]*len(nodes)))
    counter = dict(zip(nodes, [0]*len(nodes)))
    if shuffle: random.shuffle(nodes)
    while len(nodes)>0:
        k = nodes.pop(0)
        counter[k] += 1
        hist = [1e4] + [0] * n
        for p in conmap[k]:
            hist[colors[p]] += 1
        if min(hist)==0:
            colors[k] = hist.index(min(hist))
            counter[k] = 0
            continue
        hist[colors[k]] = 1e4
        minc = hist.index(min(hist))
        if counter[k]==rand:
            counter[k] = 0
            minc = random.randint(1,4)
        colors[k] = minc
        for p in conmap[k]:
            if colors[p] == minc:
                nodes.append(p)
    return colors
    
if __name__ == '__main__':
    from time import time
    from skimage.io import imread, imsave
    from skimage.data import coffee
    from skimage.segmentation import slic
    import matplotlib.pyplot as plt
    
    lab = slic(coffee(), 1000, 10, 10, 0)+1
    #lab = imread('lab.tif')
    a = time()
    idx = connect(lab, 2)
    print(time()-a)

    a = time()
    idx = connect(lab, 2)
    print(time()-a)

    a = time()
    idx = mapidx(idx)
    print(time()-a)

    a = time()
    colors = render_net(idx, 4, 10)
    lut = np.ones(lab.max()+1, dtype=np.uint8)
    for i in colors: lut[i] = colors[i]
    lut[0] = 0
    print(time()-a)

    
    plt.imshow(lut[lab])
    plt.figure()
    plt.imshow(lab)
    plt.show()