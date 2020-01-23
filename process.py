#!/usr/bin/env python3

import sys
import subprocess
import json
import base64

import cv2

import numpy as np
import matplotlib.pyplot as plt


def c2k(c):
    return c + 273.15


def k2c(c):
    return c - 273.15


def c2f(x):
  return x * 9/5. + 32.


def configure(ss):
    d = {}

    # inputs
    d['Air_Temp'] = None
    d['humidity']  = None
    d['Emissivity'] = None
    d['Refl_Temp'] = None
    d['Distance'] = None

    # constants
    d['Alpha1'] = None
    d['Alpha2'] = None
    d['Beta1'] = None
    d['Beta2'] = None
    d['X'] = None
    d['PlanckR1'] = None
    d['PlanckR2'] = None
    d['PlanckB'] = None
    d['PlanckF'] = None
    d['PlanckO'] = None

    # calculated
    d['tAtmC']  = None
    d['h2o']  = None


    try:
        d['Air_Temp'] = ss['AtmosphericTemperature']
        if 'C' in d['Air_Temp']:
            d['Air_Temp'] = c2k(float(d['Air_Temp'].split()[0]))

        d['humidity']  = ss['RelativeHumidity']
        if '%' in d['humidity']:
            d['humidity'] = float(d['humidity'].split()[0]) / 100.

        d['Emissivity'] = ss['Emissivity']

        d['Refl_Temp'] = ss['ReflectedApparentTemperature']
        if 'C' in d['Refl_Temp']:
            d['Refl_Temp'] = c2k(float(d['Refl_Temp'].split()[0]))

        d['Distance'] = ss['SubjectDistance']
        if 'm' in d['Distance']:
            d['Distance'] = float(d['Distance'].split()[0])

        # constants
        d['Alpha1'] = ss['AtmosphericTransAlpha1']
        d['Alpha2'] = ss['AtmosphericTransAlpha2']
        d['Beta1']  = ss['AtmosphericTransBeta1']
        d['Beta2']  = ss['AtmosphericTransBeta2']
        d['X'] = ss['AtmosphericTransX']
        d['PlanckR1'] = ss['PlanckR1']
        d['PlanckR2'] = ss['PlanckR2']
        d['PlanckB'] = ss['PlanckB']
        d['PlanckF'] = ss['PlanckF']
        d['PlanckO'] = ss['PlanckO']
    except KeyError as e:
        print(f'Unable to find parameter ({e}) in metadata, using defaults')

        # inputs
        d['Air_Temp'] = c2k(20)
        d['humidity']  = 0.5 # 50%
        d['Emissivity'] = 0.95
        d['Refl_Temp'] = c2k(20)
        d['Distance'] = 0.2 # m

        # constants
        d['Alpha1'] = 0.006569
        d['Alpha2'] = 0.012620
        d['Beta1']  = -0.002276
        d['Beta2']  = -0.006670
        d['X'] = 1.9
        d['PlanckR1'] = 14144.423
        d['PlanckR2'] = 0.027700923
        d['PlanckB'] = 1385.5
        d['PlanckF'] = 2.5
        d['PlanckO'] = -7494

    # calculated
    d['tAtmC'] = k2c(d['Air_Temp'])
    d['h2o'] = d['humidity'] * np.exp(1.5587 + 0.06939 * d['tAtmC'] - 0.00027816 * d['tAtmC']**2 + 0.00000068455 * d['tAtmC']**3)  #8.563981576

    print(d)

    #print(tAtmC, h2o)
    
    #Camera Temperature Range Max    : 250.0 C
    #Camera Temperature Range Min    : -20.0 C
    #Camera Temperature Max Clip     : 280.0 C
    #Camera Temperature Min Clip     : -40.0 C
    #Camera Temperature Max Warn     : 250.0 C
    #Camera Temperature Min Warn     : -20.0 C
    #Camera Temperature Max Saturated: 280.0 C
    #Camera Temperature Min Saturated: -60.0 C

    return d


def f_without_distance(x, d):
    raw_refl = d['PlanckR1'] / (d['PlanckR2'] * (np.exp(d['PlanckB'] / d['Refl_Temp']) - d['PlanckF'])) - d['PlanckO']
    ep_raw_refl = raw_refl*(1-d['Emissivity'])
    raw_obj = (x-ep_raw_refl)/d['Emissivity']
    t_obj_c = d['PlanckB']/np.log(d['PlanckR1']/(d['PlanckR2']*(raw_obj+d['PlanckO']))+d['PlanckF'])-273.15
    return t_obj_c


def f_with_distance(x, d):
   #Distance [m]
   tau = d['X'] * np.exp(-np.sqrt(d['Distance']) * (d['Alpha1'] + d['Beta1'] * np.sqrt(d['h2o']))) + (1-d['X']) * np.exp(-np.sqrt(d['Distance']) * (d['Alpha2'] + d['Beta2'] * np.sqrt(d['h2o'])))
   RAW_Atm = d['PlanckR1']/(d['PlanckR2']*(np.exp(d['PlanckB']/(d['Air_Temp']))-d['PlanckF']))-d['PlanckO']
   tau_RAW_Atm = RAW_Atm*(1-tau)
   RAW_Refl = d['PlanckR1']/(d['PlanckR2']*(np.exp(d['PlanckB']/(d['Refl_Temp']))-d['PlanckF']))-d['PlanckO']
   epsilon_tau_RAW_Refl = RAW_Refl*(1-d['Emissivity'])*tau
   RAW_Obj =  (x-tau_RAW_Atm-epsilon_tau_RAW_Refl)/d['Emissivity']/tau
   T_Obj = d['PlanckB']/np.log(d['PlanckR1']/(d['PlanckR2']*(RAW_Obj+d['PlanckO']))+d['PlanckF'])-273.15
   return T_Obj
 

#print(f_without_distance(10000))
#print(f_without_distance(25000))
#print(f_with_distance(10000))
#print(f_with_distance(25000))


def process(fn_in, fn_mask):
    aa = subprocess.run(['exiftool', '-b', '-json', fn_in], capture_output=True)
    if aa.returncode != 0:
        return None

    s = aa.stdout.strip()[1:-1]
    ss = json.loads(s)

    #print('\',\n\''.join([x for x in ss]))

    aa = ss['RawThermalImage']
    s = base64.b64decode(aa[7:])
    s = np.frombuffer(s, np.uint8)
    image = cv2.imdecode(s, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    image = image.byteswap()
    pixel = image

    d = configure(ss)

    #a = c2f(f_without_distance(pixel, d))
    a = c2f(f_with_distance(pixel, d))

    return a, ss


def compared_metadata(fns, rv):
    b = rv[0][-1]
    for i, fn in enumerate(fns):
        c = rv[i][-1]

        ignored_keys = [
'SourceFile',
'FileName',
'FileSize',
'FileModifyDate',
'FileAccessDate',
'ModifyDate',
'CreateDate',
'ImageUniqueID',
'ImageTemperatureMax',
'ImageTemperatureMin',
'RawThermalImage',
'DateTimeOriginal',
'AboveColor',
'BelowColor',
'OverflowColor',
'Isotherm1Color',
'Isotherm2Color',
'Meas1Params',
'PiPX1',
'PiPX2',
'PiPY1',
'PiPY2',
'EmbeddedImage']

        for k in sorted((set(b.keys()) | set(c.keys())) - set(ignored_keys)):
            if b[k] != c[k]:
                try:
                  print(fns[0], fns[i], k, b[k][:20], c[k][:20])
                except:
                  print(fns[0], fns[i], k, b[k], c[k])


def main():
    fns = ['FLIR0132.jpg', 'FLIR0163.jpg', 'FLIR0178.jpg', 'FLIR0184.jpg']

    rv = []
    for i, fn in enumerate(fns):
        mask_fn = ''
        a, ss = process(fn, mask_fn) 
        rv += [[i, a, ss]]
  
    compared_metadata(fns, rv) 
 
    mn = 1000
    mx = -1000
    print('id,min,mean,median,max,overall_min,overall_max')
    for i, fn in enumerate(fns):
        m = rv[i][1]

        v = [i, fn, np.min(m), np.mean(m), np.median(m), np.max(m)]

        mn = min(mn, v[2])
        mx = max(mx, v[-1])
        v += [mn, mx]

        print(','.join(map(str, v)))

    if False:
        mn = mn - (mx - mn) * .1
        mx = mx + (mx - mn) * .1

        mn = np.floor(mn)
        mx = np.ceil(mx)

    if False:
        for i in range(len(rv)):
            m = rv[i][1].flatten()
            plt.hist(m, bins=128, range=(mn, mx))
            plt.show()

    cmap = cv2.imread('pal.png')
    if cmap is not None:
        cmap = cmap[0, :, :]
        print('Read colormap with shape:', cmap.shape)

    def h(img, v):
        # Return the location of the first pixel equal to the value v.
        m = np.where(img == v)
        x, y = m[1][0], m[0][0]
        rv = (x, y, img[y, x])
        return rv

    for i, fn in enumerate(fns):
        img = rv[i][1]
 
        min_sp = h(img, img.min())
        max_sp = h(img, img.max())

        img = (img - mn) / float(mx - mn)

        sf = 4
        img = cv2.resize(img, None,fx=sf, fy=sf, interpolation = cv2.INTER_CUBIC)

        cv2.circle(img, (min_sp[0]*sf, min_sp[1]*sf), 11, 1, 2)
        cv2.circle(img, (max_sp[0]*sf, max_sp[1]*sf), 11, 0, 2)

        if cmap is not None and False:
            c = []
            xp = np.linspace(0, 1, cmap.shape[0])
            c = [np.interp(img, xp, cmap[:, j]) for j in range(3)]
            img = cv2.merge(c)
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX 
        fontScale = .6
        color = (0, 0, 0) 
        thickness = 1
        s1 = f'Range: [{min_sp[2]:.1f}, {max_sp[2]:.1f}] F'
        s2 = f'Group range: [{mn:.1f}, {mx:.1f}] F'

        textSize = cv2.getTextSize(s1, fontFace=font, fontScale=fontScale, thickness=thickness)
        if len(fns) > 1:
            textSize2 = cv2.getTextSize(s2, fontFace=font, fontScale=fontScale, thickness=thickness)
            textSize = ((max(textSize[0][0], textSize2[0][0]), max(textSize[0][1], textSize2[0][1])), max(textSize[1], textSize2[1]))

        org1 = (10, 30) 
        org2 = (10, 60) 
        x1 = min(org1[0], org2[0]) if len(fns) > 1 else org1[0]
        x2 = max(org1[0], org2[0]) if len(fns) > 1 else org1[0]
        y1 = min(org1[1], org2[1]) if len(fns) > 1 else org1[1]
        y2 = max(org1[1], org2[1]) if len(fns) > 1 else org1[1]
        s = 2
        img = cv2.rectangle(img, (x1-s, y1-s - textSize[0][1] - 2), (x2 + textSize[0][0] + s, y2 + textSize[1] + s), (255, 255, 255), -1) 

        img = cv2.putText(img, s1, org1, font,  fontScale, color, thickness, cv2.LINE_AA)
        if len(fns) > 1:
            img = cv2.putText(img, s2, org2, font,  fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.imwrite(f'{i:04}.png', img)


main()

