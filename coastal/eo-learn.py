#IMPORTS

from eolearn.core import EOTask, EOPatch, LinearWorkflow, Dependency, FeatureType

# We'll use Sentinel-2 imagery (Level-1C) provided through Sentinel Hub
# If you don't know what `Level 1C` means, don't worry. It doesn't matter.
from eolearn.io import S2L1CWCSInput
from eolearn.core import LoadFromDisk, SaveToDisk

# cloud detection
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector
from eolearn.mask import AddValidDataMaskTask

# filtering of scenes
from eolearn.features import SimpleFilterTask

# burning the vectorised polygon to raster
from eolearn.geometry import VectorToRaster


# The golden standard: numpy and matplotlib
import numpy as np

# import matplotlib TODO
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# For manipulating geo-spatial vector dataset (polygons of nominal water extent)
import geopandas as gpd

# Image manipulations
# Our water detector is going to be based on a simple threshold
# of Normalised Difference Water Index (NDWI) grayscale image
from skimage.filters import threshold_otsu

# Loading polygon of nominal water extent
import shapely.wkt
from shapely.geometry import Polygon

# sentinelhub-py package
from sentinelhub import BBox, CRS

#### Load the Polygon of nominal water extent and define a BBOX
#The BBOX defines an area of interest and will be used to create an EOPatch.

# The polygon of the dam is written in wkt format and WGS84 coordinate reference system. We are now loading this file
with open('./data/theewaterskloof_dam_nominal.wkt', 'r') as f:
    dam_wkt = f.read()

dam_nominal = shapely.wkt.loads(dam_wkt)

# inflate the BBOX
inflate_bbox = 0.1
minx, miny, maxx, maxy = dam_nominal.bounds

delx = maxx - minx
dely = maxy - miny
minx = minx - delx * inflate_bbox
maxx = maxx + delx * inflate_bbox
miny = miny - dely * inflate_bbox
maxy = maxy + dely * inflate_bbox

dam_bbox = BBox([minx, miny, maxx, maxy], crs=CRS.WGS84)

dam_bbox.geometry - dam_nominal


### Step 1: Intialize (and implement workflow specific) EOTasks
#### Create an EOPatch and add all EO features (satellite imagery data)

input_task = S2L1CWCSInput('TRUE-COLOR-S2-L1C', resx='20m', resy='20m', maxcc=0.5, instance_id=None)

add_ndwi = S2L1CWCSInput('NDWI')

#Burn in the nominal water extent. The VectorToRaster task expects the vectorised dataset in geopandas dataframe.

#crs={'init':'epsg:4326'} has to do with the way the geodataframe is initailized in the coordinate reference system
dam_gdf = gpd.GeoDataFrame(crs={'init':'epsg:4326'}, geometry=[dam_nominal])

dam_gdf.plot()



add_nominal_water = VectorToRaster(dam_gdf, (FeatureType.MASK_TIMELESS, 'NOMINAL_WATER'), values=1, raster_shape=(FeatureType.MASK, 'IS_DATA'), raster_dtype=np.uint8)



#Run s2cloudless cloud detector and filter out scenes with cloud coverage >20%
#To speed up the process the cloud detection is executed at lower resolution (160m). The resulting cloud probability map and binary mask are stored as CLP and CLM features in EOPatch.
cloud_classifier = get_s2_pixel_cloud_detector(average_over=2, dilation_size=1, all_bands=False)

cloud_detection = AddCloudMaskTask(cloud_classifier, 'BANDS-S2CLOUDLESS', cm_size_y='160m', cm_size_x='160m',
                                   cmask_feature='CLM', cprobs_feature='CLP', instance_id=None)

#Define a `VALID_DATA` layer: pixel has to contain data and should be classified as clear sky by the cloud detector (`CLM` equals 0)
def calculate_valid_data_mask(eopatch):
    return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                          np.logical_not(eopatch.mask['CLM'].astype(np.bool)))

add_valid_mask = AddValidDataMaskTask(predicate=calculate_valid_data_mask)

#Calculate fraction of valid pixels per frame and store it as `SCALAR` feature
def calculate_coverage(array):
    return 1.0 - np.count_nonzero(array) / np.size(array)


class AddValidDataCoverage(EOTask):

    def execute(self, eopatch):
        valid_data = eopatch.get_feature(FeatureType.MASK, 'VALID_DATA')
        time, height, width, channels = valid_data.shape

        coverage = np.apply_along_axis(calculate_coverage, 1, valid_data.reshape((time, height * width * channels)))

        eopatch.add_feature(FeatureType.SCALAR, 'COVERAGE', coverage[:, np.newaxis])
        return eopatch


add_coverage = AddValidDataCoverage()


class ValidDataCoveragePredicate:

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        return calculate_coverage(array) < self.threshold


remove_cloudy_scenes = SimpleFilterTask((FeatureType.MASK, 'VALID_DATA'), ValidDataCoveragePredicate(0.2))


class WaterDetector(EOTask):

    @staticmethod
    def detect_water(ndwi):
        """
        Very simple water detector based on Otsu thresholding method of NDWI.
        """
        otsu_thr = 1.0
        if len(np.unique(ndwi)) > 1:
            otsu_thr = threshold_otsu(ndwi)

        return ndwi > otsu_thr

    def execute(self, eopatch):
        water_masks = np.asarray([self.detect_water(ndwi[..., 0]) for ndwi in eopatch.data['NDWI']])

        # we're only interested in the water within the dam borders
        water_masks = water_masks[..., np.newaxis] * eopatch.mask_timeless['NOMINAL_WATER']

        water_levels = np.asarray([np.count_nonzero(mask) / np.count_nonzero(eopatch.mask_timeless['NOMINAL_WATER'])
                                   for mask in water_masks])

        eopatch.add_feature(FeatureType.MASK, 'WATER_MASK', water_masks)
        eopatch.add_feature(FeatureType.SCALAR, 'WATER_LEVEL', water_levels[..., np.newaxis])

        return eopatch


water_detection = WaterDetector()


workflow = LinearWorkflow(input_task, add_ndwi, cloud_detection, add_nominal_water, add_valid_mask, add_coverage,
                          remove_cloudy_scenes, water_detection)


time_interval = ['2016-01-01','2018-08-31']


result = workflow.execute({
    input_task: {
        'bbox': dam_bbox,
        'time_interval': time_interval
    },
})


patch = list(result.values())[-1]


from skimage.filters import sobel
from skimage.morphology import disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat


def plot_rgb_w_water(eopatch, idx):
    ratio = np.abs(eopatch.bbox.max_x - eopatch.bbox.min_x) / np.abs(eopatch.bbox.max_y - eopatch.bbox.min_y)
    fig, ax = plt.subplots(figsize=(ratio * 10, 10))

    ax.imshow(eopatch.data['TRUE-COLOR-S2-L1C'][idx])

    observed = closing(eopatch.mask['WATER_MASK'][idx, ..., 0], disk(1))
    nominal = sobel(eopatch.mask_timeless['NOMINAL_WATER'][..., 0])
    observed = sobel(observed)
    nominal = np.ma.masked_where(nominal == False, nominal)
    observed = np.ma.masked_where(observed == False, observed)

    ax.imshow(nominal, cmap=plt.cm.Reds)
    ax.imshow(observed, cmap=plt.cm.Blues)
    ax.axis('off')

    plot_rgb_w_water(patch, 0)