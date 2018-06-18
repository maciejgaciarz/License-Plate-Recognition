License Plate Recognition system written for RSM class at Science University of Wrocław

Authors:
Lin Lu
Tomasz Bartnik
Maciej Gąciarz


PROJECT INFO

1. Required libraries

Python v3.5.2

System: Windows/Linux/macOS

pip packages:

matplotlib==2.2.2
numpy==1.14.4
scikit_image==0.14.0
scikit_learn==0.19.1


3. Instructions:

- Install python 3.5.2 for you system from python main page

- Install requirements:

py -3.5 -m pip install -r /path/to/project/requirements.txt

- Launch project in any IDE

4. Folders

Car_images - has a pictures of indian cars that contain license plates. Used in testing final results of various classifiers


input - contains a set of 20x20 images of every alphabet letters and numbers extracted from images of license plates.
        Used for learning the classifiers.

output - contains a .pkl files that represent trained data for various classifiers.
        Used in main_py evaluating the pictures from car_images folder


5. Files

char_recognition.py - contain functions that load a model from file and recognize characters on the license plate using it.

char_segmentation.py - extracts characters from extracted license plate

classifiers_training.py - contains definitions of classifiers used, reads training data, trains classifiers and saves them as .pkl files

image_display.py - contains functions to display images

main.py - main file. Reads a file from car_images and uses functions from other files to extract a license plate and then a characters from a plate
After extracting the characters calls for char_recognition to get the resulting number plate as string and display it for all classifiers tested

plate_localization.py - separates image into regions, inverts theirs color, and searches all of the regions using given constraints:

- Rectangle shape

- Width is more than height

- Width is 15% to 40% of full image

- Height is 8% to 20% of full image

Returns array of suspected plates, then checks that array for plates using comparison of white and black pixels.

preprocessing.py - Contains functions that preprocess image before plate recognition
