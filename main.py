import preprocessing
import image_display
import plate_localization
import char_segmentation
import char_recognition
import matplotlib.pyplot as plt 
import os
import classifiers_training

# converting image into gray
gray_vehicle_image = preprocessing.read_convert('car_images/image2.jpg')
print('working')

# Noise Removing
# noise_removed_image = preprocessing.noise_remove(gray_vehicle_image/255)
# image_display.show_image(gray_vehicle_image,'Gray Image', noise_removed_image, 'Noise Removed')
# print('working')

# Contrast Enhancement 
# gray_vehicle_image = preprocessing.enhancement_image(noise_removed_image)
# image_display.show_image(noise_removed_image,'Noise Removed Image', gray_vehicle_image, 'Enhanced_image')
# print('working')

# Binary conversion using otsu thresholding
binary_image = preprocessing.thresholding(gray_vehicle_image)
image_display.show_image(gray_vehicle_image,'Gray Image', binary_image, 'Binary Image')
print('working')

# Finding out region that can be lisence plate
expected_plates = plate_localization.extract_plate(gray_vehicle_image, binary_image)
actual_plate = plate_localization.plate_detect(expected_plates)
plt.figure()
plt.imshow(actual_plate, cmap= 'gray')
print('working')
# Finding Characters in proposed plate
characters, column_list = char_segmentation.char_extraction(actual_plate)
print('working')

# Getting directory of models
models_dir = os.path.join(os.getcwd(), 'output')


# Character recognition and results display for each model
for name in classifiers_training.names:
    model = char_recognition.loading_model(os.path.join(models_dir, name + ".pkl"))
    resulting_number_plate = char_recognition.char_recog(model,characters, column_list)
    print('Plate number according to ' + name + ' is ' + resulting_number_plate)




