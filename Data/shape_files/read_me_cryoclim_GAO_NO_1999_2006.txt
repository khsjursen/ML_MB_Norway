Glacier Area Outline 1999-2006(GAO)

Variable 		Explanation

local_id		Glacier ID in NVE's database
name			Galcier name
GlacierComplexID	Glacier complex ID in NVE's database
GlacierComplexName	Glacier complex name
GlacierComplexCode	Glacier complex acronym
aqc_time		Date of data acquisition (YYYYMMDD) 
area_km2		Total glacier area in km2
ID			GLIMS ID for the glacier
loc_unc_x		Local (within-image) location uncertainty in metres 
loc_unc_y		Local (within-image) location uncertainty in metres
glob_unc_x		global (geographic) location uncertainty in metres
glob_unc_y		global (geographic) location uncertainty in metres
data_src		Data source (Satellite name)
proc_desc		Threshold value from image data processing 
inst_name		Satellite name
scene_id		Landsat scene number(PPPRRR) P=path, R=row
orig_id			Original ID of image (Landsat granule ID)
imgctrlon		Longitude of image centre, in decimal degrees
imgctrlat		Latitude of image centre, in decimal degrees
cloud_pct		Percent of image obscured by clouds
sun_azim		Solar azimuth
sun_elev		Solar elevation
auth_pub		Author/publisher of source map 
asof_date		Year of aerial photography
scale			Scale of source map
proj			Map projection
sheet			Sheet number for N50 maps
productID		Product ID for different periods
anlst_surn		Analyst surname
anlst_givn		Analyst given name


Abstract:
Glacier Area Outline (GAO) for mainland Norway from the period 1999-2006, using Landsat TM/ETM+ satellite images. A semi-automatic band ratio method was applied. 
Totally 12 satellite Landsat scenes from the period 1999-2006 were analysed. The orthorectification processing and quality check was carried out with 
the OrthoEngineTM software (©PCI Geomatica). Five of the 12 scenes were ordered raw from the United States Geological Survey (USGS) and were orthorectified 
by NVE using ground-control points. Typical control points used were lake edges or islands in lakes. The other scenes were already orthorectified and provided 
by Norsk Satelittdataarkiv at the center for GIS and Earth Observation (Arendal) or from the USGS. The quality of the orthorectification was tested against 
5-14 check points. For all scenes the horizontal positional accuracy (rmse) was less than one pixel (better than 30 m). After the orthorectification or quality 
control the individual channels band1, band3, band4 and band5 used in the band ratio method were exported to an ArcGIS (© ESRI) readable format (GeoTIFF) where 
further GIS-based processing was carried out. The suitability of a semi-automatic band-ratio method was applied to map glaciers in a test region in Jotunheimen, 
before the method was applied to map all glaciers in Norway. All automatically mapped snow and ice polygons were visually inspected using composites of satellite 
image bands, digital topographic maps and orthophotos where available. The polygons were manually classified as ëglaciersí, ëpossible snowfieldsí or ësnowí. 
Manual corrections for debris cover, glacier lake interfaces, clouds or cast shadow were made where necessary. Glacier complexes were divided into glacier units 
using drainage divides. Many smaller polygons which had been classified as possible snowfields due to size, shape or due to uncertainty regarding ice content 
were not assigned IDs and were therefore not included. References: Andreassen, L.M., and Winsvold, S.H. (eds.), 2012: Inventory of Norwegian glaciers. NVE Rapport 38, 
Norges Vassdrags- og energidirektorat, 236 s. Andreassen, L.M., F. Paul, A. K‰‰b and J.E. Hausberg. 2008. Landsat-derived glacier inventory for Jotunheimen, Norway, 
and deduced glacier changes since the 1930s. The Cryosphere, 2, 131ñ145. Paul, F. and L.M. Andreassen. 2009. A new glacier inventory for the Svartisen region (Norway) 
from Landsat ETM+ data: Challenges and change assessment. Journal of Glaciology, 55 (192), 607ñ618. Paul, F., L.M. Andreassen and S.H. Winsvold. 2011. 
A new glacier inventory for the Jostedalsbreen region, Norway, from Landsat TM scenes of 2006 and changes since 1966. Annals of Glaciology, 52 (59), 153ñ162.