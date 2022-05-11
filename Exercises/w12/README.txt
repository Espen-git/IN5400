IN5400, 2022 | Lab on semantic segmentation
*******************************************

This exercise is about modifying a quite simple implementation of a U-net for semantic image segmentation, as well as implementing a few basic performance metrics one often encounters when wanting to evaluate a segmentation model.

We will be working on the Oxford-IIIT Pet Dataset, containing images of pets and segmentation ground-truth masks mapping each pixel into one of three classes: background, the pet itself, and a border.

The tasks this week are meant to be quite limited in terms of workload, so do not worry if you finish it in a flash.

1. Getting to know the u-net implementation

 a. Please make yourself acquainted with the provided code.  Especially, make sure you go through every step of the implementation of the u-net.

 b. Currently, the u-net code is following that outlined in the illustration on slide 19 in the lecture notes (except for some batch-normalization).  As a result of the cropping (to avoid border effects), the final output layer has a reduced spatial extent.  A simple approach to ensure that the final (classification) layer covers the same spatial extent as that of the input is to (implicitly) pad the inputs when doing the convolutions.  That is, the convolution layers will not reduce the width and the height of the feature layers.  Modify the code to make this happen. [Note that getting a cropped output, like the code originally does, makes a lot of sense when working and training with image patches.]

 c. The current code only uses three down-pooling steps.  Modify the code so that it uses six such steps, one more than the one illustrated in the lecture slides.

2. Loss functions

 a. Let us now say that the object borders, represented by the third class (mask values == 2), has extra importance.  To reflect this, we want to double the weight of this class when calculating the loss.  Implement this.

3. Metrics

 a. Currently, we plot the loss.  Please add a plot showing the pixel accuracy, mean pixel accuracy, and mean IoU.  Please implement this "from scratch".

4. Other questions

 a. A line in the lecture slides reads "The later layers contain more information about “what”, while earlier layers more “where”".  Please explain.
