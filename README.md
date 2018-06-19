# CNN Research project

### Problem formulation:
* Given a sequence of pictures and border box of the object on the first frame draw the b.b. on the rest of the frames.
* Given all the frames and no b.b. at all, estimate the object of the interest and draw the b.b. on the rest of the frames.
* Given only the first frame, estimate the object of the interest and draw the b.b. on this frame.
* Given a sequence of the frames (from 1 to k), estimate the object of the interest and draw the b.b. on the first frame.
* Given a point, that belongs to the object of the interest and a frame(s), draw a b.b. Draw the exact boundaries of the object.
* Given a video, guess if it is from the tracking object dataset, or is it just a random video from the youtube.

#### Possible approaches:
* Used object detector, assume mostly the target is one of its outputs
* Direct BBox regression by CNN trained on sequences from benchmarks
