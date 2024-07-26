#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy

from geotiff import GeoTiff
#27.3325°N 88.6140°E gangtok coordinates


# In[83]:


list_vals =[]
for i in range(1,10):
    tiff_file = "ISRO/India_GISdata_LTAy_YearlyMonthlyTotals_GlobalSolarAtlas-v2_GEOTIFF/monthly/"
    tiff_file = tiff_file + f"PV_OUT_0{i}.tif"
    geo_tiff = GeoTiff(tiff_file)
    zarr_array = geo_tiff.read()
    list_vals.append(zarr_array[1279][2712])


# In[ ]:


print(list_vals)


# In[72]:


# ((88.6140- 66)/0.00833333333333286)


# In[ ]:


# //1279  -0.00833333333333286 2712

