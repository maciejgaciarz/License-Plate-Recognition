from skimage.measure import regionprops, label
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.filters import threshold_otsu

def extract_plate(gray_image, binary_image):
    plate_objects_cordinates = []
    plate_like_objects = []
    # plate size contraints
    # Ractangle in shape
    # width is more than height
    # width is 15% to 40% of full image
    # height is 8% to 20% of full image
    
    # this gets all the connected regions and groups them together
    label_image = label(binary_image)

    # getting the maximum width, height and minimum width and height that a license plate can be
    # shape[1] = width  shape[0] = height
    plate_dimensions = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_image, cmap="gray");
    # regionprop create list of properties of all labelled regions
    for region in regionprops(label_image):
        if region.area < 50:
            continue  # discard if too small
    
        # bounding box coordinates 
        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col
        
        # making sure region identified is license plate we assumed
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            plate_like_objects.append(binary_image[min_row:max_row,min_col:max_col])
            
            plate_objects_cordinates.append((min_row, min_col,max_row, max_col))
        
            rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
            # let's draw a red rectangle over those regions
    plt.show()    
    
    return plate_like_objects

def inverted_threshold(grayscale_image):
    threshold_value = threshold_otsu(grayscale_image) - 0.05
    return grayscale_image < threshold_value

def plate_detect(regions):
    lowest = 0
    for region in regions:
        total_white_pixels = 0
        image = inverted_threshold(sobel(region))
        height, width = region.shape
        for column in range(width):
            total_white_pixels += sum(image[:, column])
    
        if lowest == 0:
            lowest = total_white_pixels
            license_plate = region
        elif lowest > total_white_pixels:
            lowest = total_white_pixels
            license_plate = region
        
    return license_plate