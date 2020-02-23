## Writeup



---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./Project2_AdvancedLaneFinding.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Camera Calibration][./output_images/calibration1_chessboardUndist.png]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted Image][./output_images/Test3_Undist.png]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The different thresholds are implemented in code cells 9-14.   Here's an example of my output for this step.

![thresholded binary Image][./output_images/Test_3_combined.png]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[567, 470], [288, 660], [1009, 660], [716, 470]])
dst = np.float32([[280, 0], [280, 660], [1000, 660], [1000, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 567, 470      | 280, 0        | 
| 288, 660      | 280, 660      |
| 1009, 660     | 1000, 660      |
| 716, 470      | 1000, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warped straight lines][./output_images/straightLinesWarped.png]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After warped the binary thresholded image, the function `find_lane_pixels` was applied. This function fits the right and left lane pixels in a 2nd order polynomial function. To identify the starting point of the lane lines the function `find_lane_pixels` takes half of the image in y-direction and calculates a histogram of the pixels in the image. The maximum from left lane to the midpoint and the maximum from the midpoint to the left lane define the starting point of the lanes.

![Find lane pixels][./output_images/Test_3_findLanePx.png]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the radius of the curvature and the position of the vehicle with respect to the center two functions were implemented. The name of the function for the radius is `radius`. It takes the coordination of both lane lines, transform the pixels into unit meter and use the known formular for curvature introduced earlier in the course.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:

![Final result image]['./output_images/Test_3_FinalResult.png]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project used different filter technics such as Sobel, direction of the gradient, magnitude of the gradient and HLS color filter to identify the pixels on the lane lines. The result for the provided test images are robust and accurate. Nevertheless the pipeline might fail in case of objects entering the lane or in case of a road decline or increase.