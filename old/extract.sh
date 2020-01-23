#!/bin/bash

# yum install perl-Image-ExifTool ImageMagick

# from http://www.eevblog.com/forum/thermal-imaging/flir-e4-thermal-imaging-camera-teardown/msg348398/#msg348398

# extract Raw Thermal Image
exiftool -b -RawThermalImage $1 > t1.png

# swap byte order (here you have trouble)
#convert t1.png pgm:- | convert -endian lsb pgm:- t2.png
#or
#convert t1.png gray:- | convert -depth 16 -endian msb -size 320x240 gray:- t2.png
# or since IM 6.8.8 with switch png:swap-bytes
convert -define png:swap-bytes=on t1.png t2.png

# expand 16 Bit to visible range
convert t2.png -auto-level t3.png

# extract color table
exiftool FLIR0132.jpg -b -Palette > pal.raw

# swap Cb Cr
# convert for Windows: -colorspace sRGB | for MAC OSX: -colorspace RGB
convert -size 224X1 -depth 8 YCbCr:pal.raw -separate -swap 1,2 -set colorspace YCbCr -combine -colorspace RGB pal.png

# coloring the radiometric image with a color lookup table
convert t3.png pal.png -clut t4.png

# We see that we have chosen the wrong colorspace (used above RGB)
# better results: sRGB
# and expand video pal color table from [16,235] to [0,255] with -auto-level
convert -size 224X1 -depth 8 YCbCr:pal.raw -separate -swap 1,2 -set colorspace YCbCr -combine -colorspace sRGB -auto-level pals.png
convert t3.png pals.png -clut t5.png

