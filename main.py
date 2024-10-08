import cv2
import numpy as np
from matplotlib import pyplot as plt

image_paths=['1.png','2.png','3.png','4.png','5.png','6.png','7.png','8.png','9.png','10.png']
images=[]
n_houses=[]
priority_houses=[]
ratios=[]
for img in image_paths:
    #importing image
    image_path=img
    image_original= cv2.imread(image_path)
    #smoothing out the noise in image for better masking
    image=cv2.GaussianBlur(image_original, (49, 49), 0)
    #gaussian blur values from trail and error

    #convert the image to HSV for easier thresholding
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Red color threshold
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    #2 ranges as red is on both left and right side of the colour palette

    #blue color thresholds (for triangles)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    
    #green color threshold
    lower_green = np.array([35, 20, 20])
    upper_green = np.array([85, 255, 255])

    #brown color threshold
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([20, 255, 200])

    #masks for green and brown
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    #highlighting green areas with cyan
    image[green_mask > 0] = [255, 255, 0]  # Cyan green in BGR

    #highlighting brown areas with yellow
    image[brown_mask > 0] = [0, 255, 255]  # Yellow in BGR

    #converting to RGB and adding to the list to display later
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Create masks for red and blue triangles
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Create a copy of the original image to modify
    result_image = image.copy()

    yellow = np.array([0, 255, 255])
    cyan = np.array([255, 255, 0])

    yellow_mask = cv2.inRange(result_image, yellow, yellow)
    cyan_mask = cv2.inRange(result_image, cyan, cyan)

    # Perform morphological closing to fill holes
    #filling all the holes in the cyan mask
    kernel = np.ones((45, 45), np.uint8) #size from trail and error
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(cyan_mask, cmap='gray')
    #plt.show()

    #filling the holes with cyan in the image
    result_image[cyan_mask > 0] = [255, 255, 0] 
    #filling everything else with yellow
    result_image[cyan_mask == 0] = [0, 255, 255]

    #correcting the yellow mask to include the holes
    yellow_mask = cv2.inRange(result_image, yellow, yellow)
    #cv2.imwrite('no_triangles.png', result_image) #save if necessary to see masking

    # count for triangles
    red_on_brown = 0
    red_on_green = 0
    blue_on_brown = 0
    blue_on_green = 0

    #contouring for red triangles on burnt (yellow) region
    contours, _ = cv2.findContours(cv2.bitwise_and(yellow_mask,red_mask),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #looping through the contours
    for contour in contours:
        #approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        red_on_brown+=1

    #contouring for blue triangles on burnt (yellow) region
    contours, _ = cv2.findContours(cv2.bitwise_and(yellow_mask,blue_mask),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        blue_on_brown+=1

    #contouring for red triangles on green region
    contours, _ = cv2.findContours(cv2.bitwise_and(cyan_mask,red_mask),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        red_on_green+=1

    #contouring for blue triangles on green region
    contours, _ = cv2.findContours(cv2.bitwise_and(cyan_mask,blue_mask),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        blue_on_green+=1
    
    this=[red_on_brown+blue_on_brown,red_on_green+blue_on_green]
    n_houses.append(this)
    pb=2*blue_on_brown+red_on_brown
    pg=2*blue_on_green+red_on_green
    priority_houses.append([pb,pg])
    ratios.append(pb/pg)
print(n_houses)
print(priority_houses)
print(ratios)

fig, axs = plt.subplots(2, 5, figsize=(5, 4)) #kindly change figsize for better viewing

#converting from 2d array to 1d array for easier traversing
axs = axs.flatten()

#loop to display each image
for i, image in enumerate(images):
    axs[i].imshow(image)
    axs[i].axis('off')  #hide the axis for better viewing
    axs[i].set_title(f'Image {i+1}')
plt.show()
