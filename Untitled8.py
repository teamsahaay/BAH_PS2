#!/usr/bin/env python
# coding: utf-8

# # Converting the mask_image obtained from attention U-NET to a geo-tiff format

# In[ ]:


pip install rasterio


# In[ ]:


import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image


# Load the binary mask from an image file (replace './test.jpg' with your actual file)
# the output from the U-NET basically

mask = Image.open('./test.jpg')
mask = np.array(mask)
# mulitplying 255 since the pixel values range from [0,1]
mask = mask*255
# Define the geospatial reference information
left, bottom, right, top = -123.5, 37.5, -122.5, 38.5  # Bounding box coordinates from the area the user has selected prefarably rectanglur box
crs = 'EPSG:4326'  # Coordinate reference system  the user is using 

# we are transforming  the pixel coordinates to the geographic coordinates
transform = from_bounds(left, bottom, right, top, mask.shape[1], mask.shape[0])

# Define the metadata for the GeoTIFF
meta = {
    'driver': 'GTiff',
    'height': mask.shape[0],
    'width': mask.shape[1],
    'count': 1, # no of bands (1 since a single channel mask) #similarly a rgb image has 3 channels
    'dtype': mask.dtype,
    'crs': crs,
    'transform': transform
}

# Save the georeferenced binary mask as a GeoTIFF
output_filename = 'georeferenced_mask.tif'
with rasterio.open(output_filename, 'w', **meta) as dst:
    dst.write(mask, 1)  # Write the mask to the first band

print(f"Georeferenced mask saved as {output_filename}")


# # finding the outline of the masks from the tiff file so that we can get the polygons from these outlines

# In[ ]:


import numpy as np
import rasterio
import cv2
from shapely.geometry import Polygon, mapping
import geopandas as gpd

# Load the GeoTIFF binary mask
with rasterio.open('georeferenced_mask.tif') as src:
    mask = src.read(1)  # Read the first band
    transform = src.transform
    crs = src.crs

# Find contours in the binary mask
contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# arguments explained in the findContours:
#                   we are choosing cv2.RETR_LIST since we need to consider child contours as well 
#                   since we are dealing with satellite images extracting finer details are important for precise measurements 



# # extracting polygons from the contours and converting them into a shape

# In[ ]:


polygons = []
for contour in contours:
    if cv2.contourArea(contour) > 0:  # Exclude empty contours
        contour = contour.squeeze(axis=1)
        #removes the extra dimension generated in the contour
        # Transform pixelpoints of  to geographic coordinates
        transformed_contour = [rasterio.transform.xy(transform, point[1], point[0]) for point in contour]
        polygon = Polygon(transformed_contour)
        polygons.append(polygon)


# Create a GeoDataFrame to store the polygons
gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

# Save the polygons as a shapefile or GeoJSON
gdf.to_file('building_footprints.shp')  # or 'building_footprints.geojson'

print("Footprints saved as building_footprints.shp")


# # Extracting the area from shape_file using python OGR

# In[ ]:


# from open source gepo
from osgeo import ogr, gdal

# Load the shapefile
shapefile = ogr.Open('building_footprints.shp')
# we are getting  all the layers from the vector file 
layer = shapefile.GetLayer()

# Calculate the area for each feature basically each polygon
for feature in layer:
    geom = feature.GetGeometryRef()
    area = geom.GetArea()
    print(f"Area: {area} square meters")

