## CNN Research project
##### TODO
####### * Download standard tracking datasets VOT, OTB, ALOV, …[2018.06.14]
####### * Look for papers for object BBox regression
####### * Read about standard object detectors: YOLO, SSD, FasterRCNN
####### * Read the EdgeBoxes paper
####### * Read the TLD paper (Kalal, Mikolajczyk, Matas).
####### * Read about MaskRCNN (related to the pixel -> Bbox problem)


##### IDEAS
###### Possible approaches:
####### * Used object detector, assume mostly the target is one of its outputs + 
####### * Direct BBox regression by CNN trained on sequences from benchmarks


##### Problem formulation:
###### * Given a sequence of pictures and border box of the object on the first frame draw the b.b. on the rest of the frames. {, bb1} {bbi}||i=2 
###### * Given all the frames and no b.b. at all, estimate the object of the interest and draw the b.b. on the rest of the frames. {} {bbi}||i=2
###### * Given only the first frame, estimate the object of the interest and draw the b.b. on this frame. {1} {bb1}
###### * Given a sequence of the frames (from 1 to k), estimate the object of the interest and draw the b.b. on the first frame. {[1:k]} {bb1}
###### * Given a point, that belongs to the object of the interest and a frame(s), draw a b.b. Draw the exact boundaries of the object. {1, point} {bb || boundaries}
###### * Given a video, guess if it is from the tracking object dataset, or is it just a random video from the youtube. {} {YT, TB}

##### DONE
 
