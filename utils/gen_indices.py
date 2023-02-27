import numpy as np

def get_index(name,r,g,b,nir):
    eps = 1
    if name == 'ndvi':
        return (nir - r)/(nir+r)
    
    elif name == 'gli':
        return (2 * g - r - b) / (2 * g + r + b)
    
    elif name == 'cvi':
        return (nir / g) * (r / g)
    
    elif name == 'sipi':
        return (nir - b) / (nir - r)
    
    elif name == 'evi':
        return 2.5 * (nir - r) / (nir + 6 * r - 7.5 * b + 1)
    
    elif name == 'evi2':
        return 2.5 * (nir - r) / (nir + 2.4 * r + 1)
    
    elif name == 'ndwi':
        return (r - b) / (r + b)
    
    elif name == 'npcri':
        return (g - nir) / (g + nir)
    
    elif name == 'savi':
        return 1.5 * (nir - r) / (nir + r + 1)
    
    elif name == 'gndvi':
        return (nir - g) / (nir + g)
    
    else:
        raise NotImplementedError(f'Index {name} not implemented')

all_names = ['ndvi','evi','evi2','gndvi','ndwi','savi','gli','cvi','sipi','npcri']

def get_additional_indexes(arr,names=['evi','evi2','savi','gli']):

    r = arr[...,2:3]
    g = arr[...,1:2]
    b = arr[...,0:1]
    nir = arr[...,3:]

    arrs = []
    for name in names:
        arr = get_index(name,r,g,b,nir)
        arr = np.where(arr == np.inf, 100, arr)
        arrs.append(arr)
    arrs = np.concatenate(arrs,axis=2)
    return arrs