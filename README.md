# Tracking by detection

## Problem formulation

The standard object tracking task is the following: given the first frame from the sequence and the bounding box of the object of the interest - draw the bounding boxes of this object on a whole sequence. However, occlusions may (and usually do) happen through the sequence, and the biggest challenge is to keep tracking the object even though.

My problem is to track an object without information of its first bounding box. In other words - try to estimate the object(s) of the interest given only the first frame, several frames from the sequence or the whole sequence and then track it.

In my experiments, I used [YOLO](https://pjreddie.com/darknet/yolo) as detector and [VOT 2016](http://www.votchallenge.net/vot2016) dataset.


## Possible approach

If all, or *n >> 1* frames  are available:

- Run the tracker on a whole available images sequence to get all bounding boxes of the (possible) objects. 
- Tracker returns the coordinates of the bounding boxes and the softmax probabilities of belonging to each class. 
- With dynamic programming, estimate object of the interest, using **parameters** (see below):
  - For each box on a frame `k` - compute the "**transition energy**" to each of the boxes on a frame `k+1`. 
  - Repeat for all available frames. 
  - The chain of boxes with the most summary energy is estimated as the object of the interest.
  - Estimating several objects as interesting (with some probabilities) is possible as well
- If detector failed to detect at least something - **interpolate** box between available boxes.

The “**transition energy**" is dependent on the following **parameters**:

- Initial energy
- IoU between boxes
- Class correlations
- Tracker's descriptors (full implementation coming soon)
**
*Each of the parameters is normalized to give equal energy impact.*

**Initial energy** is an attempt to guess the object of interest by looking on the first frame only. By assumption the object of interest is probably located whether in the middle of the picture or centred, relatively to other objects. Thus, object that satisfies the above conditions will have greater energy and will be chosen with higher probability. The coefficient is chosen in the way that the **initial energy** has 10% impact on the resulted energy.

**Class correlations** are calculated in advance (offline) from the softmax probabilities of each bounding box for each frame from the dataset. For instance, an object which is tracked as a bicycle usually in the same time has a larger probability of being a motorcycle rather than a banana. That is - a bicycle with a motorcycle has a positive correlation, and a bicycle with a banana has negative (or, at least, low). 

That means that tracker usually confuse some classes and we should allow our algorithm to jump between objects with a positive correlation and forbid jumping to the objects with a negative one. 
I have run tracker on the whole dataset, collected the class probabilities for each bounding box and calculated the correlations using the formula:

![Pearson Correlation Coefficient](https://www.socscistatistics.com/images/pearson.png)


Example: correlation of car and some other classes:

![car correlations example](https://d2mxuefqeaa7sj.cloudfront.net/s_FB128E5C293902EB704F2E5AFBD5DAF582909820544A5CAD330E8CF077367D03_1535374782991_file.png)


Even though the correlation of *car* and *truck* are relatively small, it’s enough to conclude that classes are visually similar.

**Tracker's descriptors** are taken by removing the last fully connected layer from the tracker. The energy should be calculated as a vector distance between descriptors of each bounding box. Currently, I use distances between softmax class probabilities vectors (predicted by YOLO), but the mentioned approach should be used further.


## Tracker-based approach limits

One can calculate **the theoretical maximum** of detector-based tracker by choosing the best bounding box on each frame. By reducing the class precision threshold, we can increase the number of boxes to the almost unlimited amount - thus we have no upper bound for the algorithm performance. 

However, the detector may not be taught to detect the desired classes which are required in the dataset. Moreover, there always would br a trade-off: the more boxes are produced by the detector - the higher is the probability to jump to the wrong bounding box. 

Thus reducing the class precision threshold *and* developing a strong algorithm are required to achieve decent results.


## Evaluation

Even though I have the ground truth for the VOT dataset, it is not obvious how to evaluate the performance of the algorithm. Mean IoU can be used. However, it is not a sufficient approach as the current central task is to estimate the object of the interest. I suggest using the difference in performance between tracking with the true bounding box on a first frame and without. If it's relatively low - the object is predicted correctly. This approach is consistent if the tracker itself shows good results though. 

We can compare results to **the theoretical maximum**, however in such case we won't get good results if the object is predicted correctly, but the tracker failed to do its job. Comparing with the theoretical maximum will be relevant if the tracker's performance would be sufficient.

## 
## Results

**The theoretical maximum** that can be achieved by YOLO-based tracker with a *standard* thresholds on a VOT 2016 is **38.21%** of mean IoU with the groundtruth.

By using standard YOLO class precision threshold, equal coefficients to the parameters (that is - their impact on the result is same) I have achieved **23.50%** of the mean IoU with the ground truth without knowing the first true box and **28.45%** with that information. Reducing the threshold caused lower performance.

Even though the results are not sufficient, **76.67%** of the data samples are on a halfway to the theoretical maximum or closer. **31.67%** of the data samples reached the theoretical maximum (the difference between prediction and the theoretical maximum is less than 1%). 

Some of the data samples are hopeless (i.e., leaves, blanket) as YOLO does not contain the required classes (reducing the threshold may be a solution). 

The samples like ball1, ball2, and gymnastics4 failed with zero scores because the algorithm was unable to choose the correct object among the significant amount of other different objects on the frames.


## Examples

Even though YOLO doesn’t have “bag” class, with low precision threshold it successfully tracks bag, thinking of it as a “frisbie” or a “butterfly”. **Blue** rectangle is a true box, **red** is predicted, **grey** is other detections made by YOLO.

![bag](https://d2mxuefqeaa7sj.cloudfront.net/s_FB128E5C293902EB704F2E5AFBD5DAF582909820544A5CAD330E8CF077367D03_1535379838630_file.png)


Detector-based tracker is really good in tracking lonely cars.

![racing](https://d2mxuefqeaa7sj.cloudfront.net/s_FB128E5C293902EB704F2E5AFBD5DAF582909820544A5CAD330E8CF077367D03_1535380247691_file.png)


But absolutely hopeless in tracking leaves

![leaves](https://d2mxuefqeaa7sj.cloudfront.net/s_FB128E5C293902EB704F2E5AFBD5DAF582909820544A5CAD330E8CF077367D03_1535380328248_file.png)

## Further steps
- Tracking object **based on one given pixel** on a first frame that belongs to this object,
- estimating object(s) of the interest **in real time**, updating frame-by-frame, without the whole sequence given,
- to generalize, given a video - random from a YouTube or the object tracking database - **estimate its origin**.


## Conclusion

Tracking by detection is a quite promising approach to become the state-of-the-art tracker, as, theoretically, we are not limited in performance. 
The proposed dynamic-programming-based algorithm can be sufficient in estimating one or even several objects of the interest.


