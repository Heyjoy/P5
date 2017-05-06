**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/car0.png
[image21]: ./output_images/car1.png
[image22]: ./output_images/car2.png
[image23]: ./output_images/no_car0.png
[image24]: ./output_images/no_car1.png
[image25]: ./output_images/no_car2.png
[image3]: ./output_images/
[image4]: ./output_images/scale2.png
[image41]: ./output_images/scale1.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Project files:

* `main.py` main function, creating the vedio and working steps for this project
* `detect.py `main module, which have the car detect functions and result image creating.
* `classifier.py` include the classifier related training methods and tools
* `extraction.py` include features extraction tools and functions
* `datafield.py` include all adjustable hyperparameters
* `utils.py` include other support funtion.


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in` extraction.get_hog_features()` of the file called `extraction.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, you can clearly find out the Hog of the car are different to the Not car pic.


Car ![alt text][image2]

Car ![alt text][image21]

Car ![alt text][image22]

Not Car![alt text][image23]

Not Car![alt text][image24]

Not Car![alt text][image25]

#### 2. Explain how you settled on your final choice of HOG parameters.
The code for this step is contained in `datafield.py`
I tried various combinations of parameters and find out the `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and `orientations=9` have the best training result.

The parameters set as follow:

* Color Space：`'YCrCb'`, `'LUV'`also have similar good performance.
* Orient：`9`, tried different HOG orientations.
* Pix_per_cell: `8` , default value HOG pixels per cell
* Cell_per_block: `2` , default value HOG cells per block
* Hog_channel: `"ALL" `, can use 0, 1, 2, or "ALL", here I choose `ALL` for more features.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
The code for this step is contained in `classifier.py`
The steps for train the **Linear Support Vector Machine Classifier**

1. Read in all the images. 8792 car samples , 8968 non-car samples.
2. extract `spatial features`, `hist features`, `hog features` features form the image list.
3. split randomly the dataset to train samples and test samples. here I choose `train:test` as `4:1` get test accuracy 0.99. actually even I use `1:9`, the program can get accuracy 0.98.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
The code for this step is contained in `detect.py`.

The silder window search, I tried the `perspective and vanishing point concepts`
,but maybe the way I implemented is not so right, the processing speed are very slowa around `2 s/it`. The reslut are also similar as the `Hog Sub-sampling Window Search concept`.

After these try, I choose the `Hog Sub-sampling Window Search concept` method, you can find the `find_car() `at `detect.py`.
And for this concept, I setup two different scales: `1`, `1.5`

|Searches| scales | ystart | ystop | overlapping |
|:-------| :------| :----- |:----|:------|
|Search1 | 1.5    | 400 | 656|50%|
|Search2 | 1      | 400 | 496| 50%|
|...|...|...|...|...|

actually here can use more different scales hot window search, but it will slow done the processing speed. Here has `2.5 it/s`, 4 times faster then pervious method.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

Try with scale 1.5 ![alt text][image4]

Try with scale 1 ![alt text][image41]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_solution.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
this code are implemented at `detect.heatmap()`in `detect.py.`

what I do is:

1. Record the positions of positive detections in each frame of the video.  
2. From the positive detections I created a heatmap
3. then threshold that map to identify vehicle positions.
4. I pushed this value to a deque buffer, which have 10 depth
5. calculate the sum value.
6. threshold this sum value again with a litter big value, to avoid single false point on single frame.
7. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  
8. I then assumed each blob corresponded to a vehicle, constructed bounding boxes to cover the area of each blob detected.  
9. print some values on the frame for later diagnostic.

#### Below are a few examples of heatmap and detected-boxes:
![alt text][image5]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As I mentioned above, I spend a lot time on the search window implementation.
and later spend a lot time for implement a good filter and heatmap.

For the white car in video, some time in the sunshine can not detect well.

There may have two reason for it:
* First is the classifier can not detect this shape with similar color at that time well.
* Secondly even it can detect but with a few positive point of this car, the filter will also cut it out. if we lower the threshold, a false point will appear.

Therefor what it can be improve in further:
  1. introduce a CNN for more better classifier
  2. use more different parameters for hotwindows create, and accordingly change the threshold value.
