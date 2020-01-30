#!/usr/bin/env python3

import base64
import json
import os
import subprocess
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pyemd import emd
from sklearn.metrics.pairwise import euclidean_distances

palette_f = ['exif', 'file', 'hardcoded'][0]


def c2k(c):
    return c + 273.15


def k2c(c):
    return c - 273.15


def c2f(x):
    return x * 9 / 5. + 32.


warn_distance = True


def configure(ss):
    d = {}

    # inputs
    d['Air_Temp'] = None
    d['humidity'] = None
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
    d['tAtmC'] = None
    d['h2o'] = None

    try:
        d['Air_Temp'] = ss['AtmosphericTemperature']
        if 'C' in d['Air_Temp']:
            d['Air_Temp'] = c2k(float(d['Air_Temp'].split()[0]))

        d['humidity'] = ss['RelativeHumidity']
        if '%' in d['humidity']:
            d['humidity'] = float(d['humidity'].split()[0]) / 100.

        d['Emissivity'] = ss['Emissivity']

        d['Refl_Temp'] = ss['ReflectedApparentTemperature']
        if 'C' in d['Refl_Temp']:
            d['Refl_Temp'] = c2k(float(d['Refl_Temp'].split()[0]))

        try:
            d['Distance'] = ss['SubjectDistance']
        except KeyError:
            try:
                d['Distance'] = ss['ObjectDistance']
            except KeyError:
                d['Distance'] = ss['FocusDistance']
        if 'm' in d['Distance']:
            d['Distance'] = float(d['Distance'].split()[0])
        if d['Distance'] == 0:
            d['Distance'] = 0.2
            global warn_distance
            if warn_distance:
                print('Warning: Distance = 0., increasing to 0.2, disabling warning')
                warn_distance = False

        # constants
        d['Alpha1'] = ss['AtmosphericTransAlpha1']
        d['Alpha2'] = ss['AtmosphericTransAlpha2']
        d['Beta1'] = ss['AtmosphericTransBeta1']
        d['Beta2'] = ss['AtmosphericTransBeta2']
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
        d['humidity'] = 0.5  # 50%
        d['Emissivity'] = 0.95
        d['Refl_Temp'] = c2k(20)
        d['Distance'] = 0.2  # m

        # constants
        d['Alpha1'] = 0.006569
        d['Alpha2'] = 0.012620
        d['Beta1'] = -0.002276
        d['Beta2'] = -0.006670
        d['X'] = 1.9
        d['PlanckR1'] = 14144.423
        d['PlanckR2'] = 0.027700923
        d['PlanckB'] = 1385.5
        d['PlanckF'] = 2.5
        d['PlanckO'] = -7494

    # calculated
    d['tAtmC'] = k2c(d['Air_Temp'])
    d['h2o'] = d['humidity'] * np.exp(1.5587 + 0.06939 * d['tAtmC'] - 0.00027816 * d['tAtmC'] ** 2 + 0.00000068455 * d['tAtmC'] ** 3)  # 8.563981576

    print(d)

    return d


def f_without_distance(x, d):
    raw_refl = d['PlanckR1'] / (d['PlanckR2'] * (np.exp(d['PlanckB'] / d['Refl_Temp']) - d['PlanckF'])) - d['PlanckO']
    ep_raw_refl = raw_refl * (1 - d['Emissivity'])
    raw_obj = (x - ep_raw_refl) / d['Emissivity']
    t_obj_c = d['PlanckB'] / np.log(d['PlanckR1'] / (d['PlanckR2'] * (raw_obj + d['PlanckO'])) + d['PlanckF']) - 273.15
    return t_obj_c


def f_with_distance(x, d):
    # Distance [m]
    tau = d['X'] * np.exp(-np.sqrt(d['Distance']) * (d['Alpha1'] + d['Beta1'] * np.sqrt(d['h2o']))) + (1 - d['X']) * np.exp(
        -np.sqrt(d['Distance']) * (d['Alpha2'] + d['Beta2'] * np.sqrt(d['h2o'])))
    RAW_Atm = d['PlanckR1'] / (d['PlanckR2'] * (np.exp(d['PlanckB'] / (d['Air_Temp'])) - d['PlanckF'])) - d['PlanckO']
    tau_RAW_Atm = RAW_Atm * (1 - tau)
    RAW_Refl = d['PlanckR1'] / (d['PlanckR2'] * (np.exp(d['PlanckB'] / (d['Refl_Temp'])) - d['PlanckF'])) - d['PlanckO']
    epsilon_tau_RAW_Refl = RAW_Refl * (1 - d['Emissivity']) * tau
    RAW_Obj = (x - tau_RAW_Atm - epsilon_tau_RAW_Refl) / d['Emissivity'] / tau
    T_Obj = d['PlanckB'] / np.log(d['PlanckR1'] / (d['PlanckR2'] * (RAW_Obj + d['PlanckO'])) + d['PlanckF']) - 273.15
    return T_Obj


def process(fn_in, fn_mask):
    aa = subprocess.run(['exiftool', '-b', '-json', fn_in], capture_output=True)
    if aa.returncode != 0:
        return None

    s = aa.stdout.strip()[1:-1]
    metadata = json.loads(s)

    # print('\',\n\''.join([x for x in ss]))

    aa = metadata['RawThermalImage']
    s = base64.b64decode(aa[7:])
    s = np.frombuffer(s, np.uint8)
    image = cv2.imdecode(s, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    image = image.byteswap()

    palette = None
    if palette_f == 'exif':
        n_pal_colors = metadata['PaletteColors']
        aa = metadata['Palette']
        s = base64.b64decode(aa[7:])
        s = np.frombuffer(s, np.uint8)
        s.shape = (n_pal_colors, 1, 3)
        palette = cv2.cvtColor(s, cv2.COLOR_YCrCb2BGR)
        palette.shape = (n_pal_colors, 3)

    thermo_calc_params = configure(metadata)

    # img = c2f(f_without_distance(image, thermo_calc_params))
    img = c2f(f_with_distance(image, thermo_calc_params))

    mask = None
    if fn_mask:
        mask = cv2.imread(fn_mask, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print('Unable to read mask:', fn_in)
        else:
            mask2 = np.ones(mask.shape[:2], np.float)
            mask2[mask[:, :, 3] == 0] = 0
            # mask2[mask3[:, :] == 0] = 0

            if True:
                # Don't confuse the edge created by a resize (increase in size) as a border that should be removed.
                # Look at the 1:1 images to determine if there is a need to erode the edges.

                n1 = int(np.sum(mask2))

                mask3 = np.zeros(mask.shape[:2], np.uint8)
                mask3[mask[:, :, 3] != 0] = 255

                # A (3,3) cross removes roughly one pixel width
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                # A (3,3) rectangle removes roughly two pixel width
                # kernel = np.ones((3, 3), np.uint8)

                mask3 = cv2.erode(mask3, kernel, iterations=1)
                mask2[mask3 == 0] = 0

                n2 = int(np.sum(mask2))
                print(f'Erosion pixel count reduction: {n1} - {n2} = {n1 - n2}')

            mask = mask2

    return img, mask, metadata, palette


def load_images(fns):
    rv = []
    palette = None
    for i, fn in enumerate(fns):
        mask_fn = os.path.join(os.path.dirname(fn), 'masks', os.path.basename(fn).replace('.jpg', '.png'))
        if not os.path.exists(mask_fn):
            mask_fn = ''
        img, mask, metadata, palette = process(fn, mask_fn)
        rv += [[i, img, mask, metadata]]

    return rv, palette


def compared_metadata(fns, rv):
    b = rv[0][-1]
    for i, fn in enumerate(fns):
        c = rv[i][-1]

        ignored_keys = ['SourceFile',
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
                        'EmbeddedImage',
                        'RawValueMedian',
                        'RawValueRange',
                        'ThumbnailLength',
                        'ThumbnailImage'
                        ]

        for k in sorted((set(b.keys()) | set(c.keys())) - set(ignored_keys)):
            if b[k] != c[k]:
                try:
                    print(fns[0], fns[i], k, b[k][:20], c[k][:20])
                except:
                    print(fns[0], fns[i], k, b[k], c[k])


def find_range(fns, rv):
    mn = 1000
    mx = -1000
    print('id, filename, min, mean, median, max, overall_min, overall_max')
    for i, fn in enumerate(fns):
        img, mask = rv[i][1:3]
        if mask is not None:
            img = img[mask == 1]

        v = [i, fn, np.min(img), np.mean(img), np.median(img), np.max(img)]

        mn = min(mn, v[2])
        mx = max(mx, v[-1])
        v += [mn, mx]

        print(', '.join(map(str, v)))

    if False:
        mn = mn - (mx - mn) * .1
        mx = mx + (mx - mn) * .1

        mn = np.floor(mn)
        mx = np.ceil(mx)

    return mn, mx


def create_image(img, mask, mn, mx, fns, palette):
    def h(img, v):
        # Return the location of the first pixel equal to the value v.
        if mask is not None:
            m = np.where(np.multiply(img, mask) == v)
        else:
            m = np.where(img == v)
        x, y = m[1][0], m[0][0]
        rv = (x, y, img[y, x])
        return rv

    if mask is not None:
        min_sp = h(img, img[mask == 1].min())
        max_sp = h(img, img[mask == 1].max())
    else:
        min_sp = h(img, img.min())
        max_sp = h(img, img.max())

    img = (img - mn) / float(mx - mn)

    sf = 4
    if sf != 1:
        if mask is not None:
            # When scaling a masked image, cubic interpolation will create very misleading false edge artifacts at masked edges.
            # Nearest also does but to a lesser extent. Linear interpolation is the winner when masking.
            # The artifacts come from interpolating measurements with zero values from areas masked off.
            # Masking was moved to after scaling, which requires resizing the mask. This may only shift the problem around.

            # img = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_LINEAR)
        else:
            # When not using a mask, use cubic interpolation, which seems to generate slightly sharper images.
            img = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_CUBIC)

        if mask is not None:
            mask = cv2.resize(mask, None, fx=sf, fy=sf, interpolation=cv2.INTER_LINEAR)

    if mask is not None:
        # This could be a resized mask, which will have a continuum of values [0, 1] instead of only 0 or 1.
        # So, the comparison must be made against 1, which are closest to values from the original mask.
        # There's probably some safe non-one threshold that could be used. Close enough.
        img[mask != 1] = 0

    if True:
        # Highlight the minimum and maximum values
        cv2.circle(img, (min_sp[0] * sf, min_sp[1] * sf), 11, 1, 2)
        cv2.circle(img, (max_sp[0] * sf, max_sp[1] * sf), 11, 0, 2)

    draw_palette_f = True
    palette_position = (10, 80)
    palette_position_size = (20, 256)

    if draw_palette_f:
        # Draw a palette placeholder

        for i in range(palette_position_size[1]):
            x, y = palette_position
            y += (palette_position_size[1] - i - 1)
            img = cv2.line(img, (x, y), (x + palette_position_size[0], y), i / (palette_position_size[1] - 1.))

    if palette_f == 'hardcoded' or palette is None:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        # img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
        img = cv2.applyColorMap(img, cv2.COLORMAP_PLASMA)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        xp = np.linspace(0, 1, palette.shape[0])
        c = [np.interp(img, xp, palette[:, j]) for j in range(3)]
        img = cv2.merge(c)
        img = np.clip(img, 0, 255).astype(np.uint8)

    if draw_palette_f:
        # Outline the palette

        x1, y1 = palette_position
        x2, y2 = x1 + palette_position_size[0], y1 + palette_position_size[1]
        img = cv2.rectangle(img, (x1, y1-1), (x2, y2), (255, 255, 255), 1)

    if draw_palette_f:
        # Draw the legend text on the palette

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .6
        color = (0, 0, 0)
        thickness = 1
        vals = [min_sp[2], max_sp[2], mn, mx]
        lbls = [f'{x:.1f} F' for x in vals]

        textSize = ((0,0), 0)
        szs = []
        for i in range(len(lbls)):
            sz = cv2.getTextSize(lbls[i], fontFace=font, fontScale=fontScale, thickness=thickness)
            szs += [sz]
            textSize = ((max(textSize[0][0], sz[0][0]), max(textSize[0][1], sz[0][1])), max(textSize[1], sz[1]))

        s = 1
        os = 4

        for i, (lbl, sz, val) in enumerate(zip(lbls, szs, vals)):
            if i == 2:
                x, y = palette_position[0] + 1, palette_position[1] + palette_position_size[1] + sz[0][1] + os + textSize[0][1]/2 + 0
            elif i == 3:
                x, y = palette_position[0] + 1, palette_position[1] - os - textSize[0][1]/2 - 2
            elif i == 0: # and (abs(vals[0] - vals[2]) > .001):
                v = 1 - (val - mn) / (mx - mn)
                x, y = palette_position[0] + palette_position_size[0] + os, palette_position[1] + sz[0][1] / 2 + palette_position_size[1] * v - 1
            elif i == 1: # and (abs(vals[1] - vals[3]) > .001):
                v = 1 - (val - mn) / (mx - mn)
                x, y = palette_position[0] + palette_position_size[0] + os, palette_position[1] + sz[0][1] / 2 + palette_position_size[1] * v - 1
            else:
                continue

            x, y = int(round(x)), int(round(y))

            img = cv2.rectangle(img, (x - s, y - s - sz[0][1]), (x + sz[0][0] + s, y + s + 1), (255, 255, 255), -1)
            img = cv2.putText(img, lbl, (x,y), font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        # Draw the legend text

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

        s = 2
        org1 = (10, 30)
        org2 = (org1[0], org1[1] + textSize[0][1] + textSize[1] + s)

        x1, x2 = org1[0], org1[0]
        y1, y2 = org1[1], org1[1]
        if len(fns) > 1:
            x1, x2 = min(x1, org2[0]), max(x2, org2[0])
            y1, y2 = min(y1, org2[1]), max(y2, org2[1])
        img = cv2.rectangle(img, (x1 - s, y1 - s - textSize[0][1] - 2), (x2 + textSize[0][0] + s, y2 + textSize[1] + s), (255, 255, 255), -1)

        img = cv2.putText(img, s1, org1, font, fontScale, color, thickness, cv2.LINE_AA)
        if len(fns) > 1:
            img = cv2.putText(img, s2, org2, font, fontScale, color, thickness, cv2.LINE_AA)

    return img


def perform_analysis(rv, fns, mn, mx):
    hist = []
    nbins = 128

    b = []
    for i, fn in enumerate(fns):
        img, mask = rv[i][1:3]
        if mask is not None:
            img = img[mask == 1]

        # 128 bins with a consistent range
        h, b = np.histogram(img, bins=nbins, range=(mn, mx))

        # Estimate the cumulative distribution function
        # Use cumulative sum to smooth the noise present in a histogram of quantized data
        cdf = np.cumsum(h * np.diff(b))

        # Normalize, removing effect of size
        cdf /= np.sum(cdf)

        hist += [cdf]

    rv2 = np.zeros([len(rv), len(rv)])

    bin_centers = (b[1:] + b[:-1]) / 2
    # w = euclidean_distances(np.arange(nbins).reshape(-1, 1), np.arange(nbins).reshape(-1, 1)) / (nbins - 1.)
    w = euclidean_distances(bin_centers.reshape(-1, 1), bin_centers.reshape(-1, 1))

    for i, fn_i in enumerate(fns):
        print(f'{i + 1}/{len(fns)}...')
        a = hist[i]
        for j, fn_j in enumerate(fns):
            b = hist[j]
            v = emd(a, b, w)
            rv2[i, j] = v
    np.savez('emd_results.npz', rv2)
    # np.savetxt('emd_results.csv', rv2, delimiter=',')
    print('To plot the EMD results run ./old/plot_emd_mat.py')


def filter_filenames(fns):
    fns2 = []
    for fn in fns:
        if not os.path.isfile(fn):
            print('Warning: Ignoring non-file:', fn)
        elif os.path.splitext(fn)[1].lower() not in ['.jpg', '.jpeg']:
            print('Warning: Ignoring non-jpeg file:', fn)
        elif not os.access(fn, os.R_OK):
            print('Warning: Ignoring inaccessible file:', fn)
        else:
            fns2 += [fn]

    return fns2


def main():
    if len(sys.argv) <= 1:
        print(f'Usage: {sys.argv[0]} <filename> ...')
        sys.exit(1)

    fns = sys.argv[1:]
    fns = filter_filenames(fns)
    print(fns)

    rv, palette = load_images(fns)

    compared_metadata(fns, rv)

    mn, mx = find_range(fns, rv)

    if False:
        for i in range(len(rv)):
            img, mask = rv[i][1:3]
            if mask is not None:
                m = img[mask == 1]
            else:
                m = img.flatten()
            plt.hist(m, bins=128, range=(mn, mx))
            plt.show()

    if palette_f == 'file':
        palette = cv2.imread('pal.png')
        if palette is not None:
            palette = palette[0, :, :]
            print('Read colormap with shape:', palette.shape)

    for i, fn in enumerate(fns):
        img, mask = rv[i][1:3]
        img = create_image(img, mask, mn, mx, fns, palette)
        if img is None:
            print(f'Unable to process {fn}')
            continue

        # fn2 = f'{i:04}.png'
        fn2 = os.path.splitext(os.path.basename(fn))[0] + '.png'

        cv2.imwrite(fn2, img)

    # perform_analysis(rv, fns, mn, mx)


main()

# print(tAtmC, h2o)

# Camera Temperature Range Max    : 250.0 C
# Camera Temperature Range Min    : -20.0 C
# Camera Temperature Max Clip     : 280.0 C
# Camera Temperature Min Clip     : -40.0 C
# Camera Temperature Max Warn     : 250.0 C
# Camera Temperature Min Warn     : -20.0 C
# Camera Temperature Max Saturated: 280.0 C
# Camera Temperature Min Saturated: -60.0 C

# print(f_without_distance(10000))
# print(f_without_distance(25000))
# print(f_with_distance(10000))
# print(f_with_distance(25000))
