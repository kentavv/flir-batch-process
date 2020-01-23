# flir-batch-process
Batch process FLIR radiometric (thermal) JPEGs and generate clean lossless images with consistent scale and palette

Given several FLIR radiometric JPEGs, find the temperature range across all images, use the same scale 
and palette for all images, and write lossless final images with minimal chart junk.

Originally created to batch process thermal images of plants for a plant phenotyping study, 
the script was later made more generic to process thermal images of a spindle on a CNC mill. 

The processing script relies upon ExifTool to extract the EXIF data from FLIR radiometric JPEGs. 
The script has been developed with a FLIR E5 infrared camera.

This work is indebted to many on the EEVBlog and ExifTool forums who figured out the file format and conversion
equations from raw values to temperatures.
