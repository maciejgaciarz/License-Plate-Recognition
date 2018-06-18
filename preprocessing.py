from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu           
from skimage.filters import median
from skimage.morphology import disk
from skimage.exposure import rescale_intensity
def read_convert(image_name):
    # reading Image
    vehicle_image = imread(image_name)
    gray_image = rgb2gray(vehicle_image)
    
    # skimage change into gray using 0 to 1 scale 
    # so for convinence we extend range back to 0 to 255
    gray_car_image = gray_image * 255
    
    return gray_car_image


def thresholding(gray_image):
    thresh_value = threshold_otsu(gray_image) 
    binary_image = gray_image > thresh_value
    
    return binary_image

def noise_remove(image_name):
    return median(image_name, disk(5))

def enhancement_image(image_name):
    return rescale_intensity(image_name)