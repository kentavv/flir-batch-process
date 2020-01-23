#!/usr/bin/python3

import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics.pairwise import euclidean_distances
from pyemd import emd

#fn_in = sys.argv[1]
#fn_mask = sys.argv[2]

# inputs
Air_Temp=293.15 # K
humidity =0.5 # 50%
Emissivity=0.95
Refl_Temp=303.15
Distance=100 # m

Air_Temp=293.15 # 20 C in K
humidity =0.5 # 50%
Emissivity=0.95
#Refl_Temp=303.15
Refl_Temp=293.15 # 20 C in K
Distance=0.2 # m

# constants
Alpha1=0.006569
Alpha2=0.01262
Beta1 =-0.002276
Beta2=-0.00667
X=1.9
PlanckR1=14168.402
PlanckR2=0.026881648
PlanckB=1386
PlanckF=2.5
PlanckO=-7363

Alpha1=0.006569
Alpha2=0.012620
Beta1 =-0.002276
Beta2 =-0.006670
X=1.9
PlanckR1=14144.423
PlanckR2=0.027700923
PlanckB=1385.5
PlanckF=2.5
PlanckO=-7494

# calculated
tAtmC=Air_Temp - 273.15 # 20 C
h2o=humidity * np.exp(1.5587 + 0.06939 * tAtmC - 0.00027816 * tAtmC**2 + 0.00000068455 *tAtmC**3)  #8.563981576

#print(tAtmC, h2o)

#Camera Temperature Range Max    : 250.0 C
#Camera Temperature Range Min    : -20.0 C
#Camera Temperature Max Clip     : 280.0 C
#Camera Temperature Min Clip     : -40.0 C
#Camera Temperature Max Warn     : 250.0 C
#Camera Temperature Min Warn     : -20.0 C
#Camera Temperature Max Saturated: 280.0 C
#Camera Temperature Min Saturated: -60.0 C

def f_without_distance(x):
    raw_refl = PlanckR1 / (PlanckR2 * (np.exp(PlanckB / Refl_Temp) - PlanckF)) - PlanckO
    ep_raw_refl = raw_refl*(1-Emissivity)
    raw_obj = (x-ep_raw_refl)/Emissivity
    t_obj_c = PlanckB/np.log(PlanckR1/(PlanckR2*(raw_obj+PlanckO))+PlanckF)-273.15
    return t_obj_c

def f_with_distance(x):
   #Distance [m]
   tau = X * np.exp(-np.sqrt(Distance) * (Alpha1 + Beta1 * np.sqrt(h2o))) + (1-X) * np.exp(-np.sqrt(Distance) * (Alpha2 + Beta2 * np.sqrt(h2o)))
   RAW_Atm = PlanckR1/(PlanckR2*(np.exp(PlanckB/(Air_Temp))-PlanckF))-PlanckO
   tau_RAW_Atm = RAW_Atm*(1-tau)
   RAW_Refl = PlanckR1/(PlanckR2*(np.exp(PlanckB/(Refl_Temp))-PlanckF))-PlanckO
   epsilon_tau_RAW_Refl = RAW_Refl*(1-Emissivity)*tau
   RAW_Obj =  (x-tau_RAW_Atm-epsilon_tau_RAW_Refl)/Emissivity/tau
   T_Obj = PlanckB/np.log(PlanckR1/(PlanckR2*(RAW_Obj+PlanckO))+PlanckF)-273.15
   return T_Obj
 
def c2f(x):
  return x * 9/5. + 32.


#print(f_without_distance(10000))
#print(f_without_distance(25000))
#print(f_with_distance(10000))
#print(f_with_distance(25000))

def process(fn_in, fn_mask):
  image = PIL.Image.open(fn_in)
  pixel = np.array(image)

#print(pixel)

#print("PIL:", pixel.dtype)
#print(pixel)
#print(max(max(row) for row in pixel))

#print((pixel / (2**16 - 1) * (250 - -20) + -20)  * 9/5. + 32)
#print(c2f(f_without_distance(pixel[0, 0])))
#print(c2f(f_with_distance(pixel[0, 0])))

  a = c2f(f_without_distance(pixel))
  #a = c2f(f_with_distance(pixel))

  image = PIL.Image.open(fn_mask)
  pixel = np.array(image)

  aa = pixel[:,:,3]
  aa=aa.flatten()
  b = a.flatten()[aa == 255] 

  return b

def main():
    a = list(range(2, 10+1))
    a += list(range(12, 20+1))

    a += list(range(22, 30+1))
    a += list(range(32, 40+1))

    a += list(range(43, 51+1))
    a += list(range(53, 61+1))

    a += list(range(63, 71+1))
    a += list(range(72, 80+1))

    rv = []
    for i in a:
        in_fn = 'FLIR{0:04d}.png'.format(i)
        mask_fn = 'mask/FLIR{0:04d}.png'.format(i)
        a = process(in_fn, mask_fn) 
        #print('{0:04d}'.format(i), a.shape)
        rv += [[i, list(a)]]
    
    #plt.hist(a.flatten(), bins='auto')
    #for i in range(len(rv)):
    #  plt.hist(rv[i][1], bins='auto')
    #  plt.show()

    lst = []
    print('id,min,mean,median,max')
    for i in range(len(rv)):
        m = rv[i][1]
        v = [rv[i][0], np.min(m), np.mean(m), np.median(m), np.max(m)]
        lst += [v]
        print(','.join(map(str, v)))
    #print(lst)
#    exit(1)

    #print(rv[0])

    mn = 1000
    mx = -1000
    for i in range(len(rv)):
        mn = min([mn] + rv[i][1])
        mx = max([mx] + rv[i][1])

    mn = np.floor(mn)
    mx = np.ceil(mx)

    print(mn, mx)

    for i in range(len(rv)):
      #h, b = np.histogram(rv[i][1], bins=128, range=(mn, mx))
      plt.hist(rv[i][1], bins=128, range=(mn, mx))
      plt.show()

    for i in range(len(rv)):
        # 128 bins with a consistent range
        h, b = np.histogram(rv[i][1], bins=128, range=(mn, mx))
        #print(h, b)

        # estimate the cumulative distribution function
        # use cummulative sum to smooth the noise present in a histogram of quantized data
        cdf = np.cumsum(h * np.diff(b))
        # normalize, removing effect of size
        cdf /= np.sum(cdf)
        #histo[k] = cdf
        #print(cdf)
        rv[i] = rv[i] + [cdf]

    print(rv[i])

   
    #rv2 = np.zeros([len(rv)+1, len(rv)+1])
    rv2 = np.zeros([len(rv), len(rv)])
    for i in range(len(rv)):
        for j in range(len(rv)):
            a = rv[i][2]
            b = rv[j][2]
            w = euclidean_distances(a.reshape(-1, 1), b.reshape(-1, 1))
            v = emd(a, b, w)
            v *= 10000
            print(i, j, v)
            #rv2[i+1, 0] = rv[i][0]
            #rv2[0, j+1] = rv[j][0]
            #rv2[i+1, j+1] = v
            rv2[i, j] = v
    np.savez('out.npz', rv2)
    np.savetxt('out.csv', rv2, delimiter=',')


main()

