# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Step 1: Read image and convert read image to grayscale
* Step 2: Apply Gaussian Blur function
* Step 3: Apply Canny edge detection function
* Step 4: Define region of interest
* Step 5: Apply Hough Transformation
* Step 6: Draw left and right line approximation based on Hough lines on original image

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied the gaussian blur function to be
able to use the canny function for edge detection. Next, the region of interest was defined and the Hough Transformation
was performed in the region of interest. With these information the slope and interception for the left and right line could be derived, these on the other hand, represent, the left and right lane markings of the road

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by the following steps

* Step 1: Intialised needed variables (left slope, left interception, right slope, right intersection, y max of the pic and y min)
* Step 2: Identify slope and intersection in each line in lines and update y min if necessary 
* Step 3: Assign values(slope and interception) if slope <0 to the right line and if slope > 0 to the left line and append it
* Step 4: After going through all lines, calculate median of slope, interception, x min and x max
* Step 5: Draw the lines in intialised empty image

### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be what would happen when the edge detections are outside of the set parameter. No lane markings
would be detected

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to automatise an optimization for the parameter used.

