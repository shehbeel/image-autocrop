# IMAGE AUTOCROP
# Author: Shehbeel Arif
# Description: Function to Autocrop an image to include only main object
# Code modified from: https://youtu.be/IBQYqwq_w14

# Dependencies
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Autocrop Function
def autocrop(image, lower, higher):
    # Convert image form BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make copy of image
    copy_image = image.copy()

    # Place mask on image
    lower = np.array([lower,lower,lower])
    higher = np.array([higher,higher,higher])
    mask = cv2.inRange(image, lower, higher)

    # Find the contours of image
    cont,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    # Image with contours on it
    cont_image = cv2.drawContours(image,cont,-1,255,3)

    # Find the maximum contour area
    c = max(cont,key=cv2.contourArea)

    # Make a bounding box around this contour area and extract dims
    x,y,w,h = cv2.boundingRect(c)

    # Place bounding box on image
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)

    # Crop image based on bounding box
    cropped_image = copy_image[y:y+h,x:x+w]
    
    # Display annotated, original, and cropped images
    #plt.subplot(1,3,1), plt.imshow(image)
    #plt.subplot(1,3,2), plt.imshow(copy_image)
    #plt.subplot(1,3,3), plt.imshow(cropped_image)
    
    # Save image
    #cv2.imwrite(os.path.join(out_directory, 'image.png'), cropped_image) # Add out_directory as input
    
    # Output cropped image
    return cropped_image


###################################################
#################Test the Function#################


### Autocropping a single image
# Load image
image = cv2.imread('/.../image.png')
# Autocrop the image
cropped_image = autocrop(image,20,250)
# Look at the cropped image
plt.imshow(cropped_image)


### Autocropping multiple images in a directory and saving them in new directory

# Input Images Directory
img_dir = '.../images-dir'
# Output Cropped Images Directory
out_dir = '.../cropped-images-dir'

# Initialize counter
i=1
# For loop to iterate over Input Images Directory
for img in os.listdir(img_dir):
    # Load image
    image = cv2.imread(img_dir + img)
    # Autocrop image
    cropped_img = autocrop(image,20,250)
    # Save cropped images in Cropped Images Directory
    cv2.imwrite(os.path.join(out_dir, 'image' + str(i) + '.png'), cropped_img)
    # Update counter
    i+=1

