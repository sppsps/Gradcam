# TENSORFLOW IMPLEMENTATION OF GRADCAM

## usage
```
python gradcam.py
```
## Help Log
```
Usage: ipykernel_launcher.py [-h] [--img_src IMAGE SOURCE]
                               
optional arguments:
  -h, --help            show this help message and exit
  --img_src             IMAGE SOURCE
  
```

## Contributers:
- [Pranjal Sharma](https://github.com/sppsps)
- [Dhrubajit Basumatary](https://github.com/dhruvz9)

## REFERENCE
 - Title : Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization<br />
 - Link : https://arxiv.org/abs/1610.02391 <br />
 - Author : Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra <br />
 - Published : 3 Dec 2019 <br />

# Summary

## Implementation
Grad-Cam uses the gradient information flowing into the last convolutional layer of the CNN to understand each neuron for a decision of interest.


Let Ak be the kth feature map (k=1,⋯,K) from the last convolusional layer. Grad-CAM utilize these Ak to visualize the decision made by CNN.

* Visualization of the final feature map (Ak) will show the discriminative region of the image. <br />
* Simplest summary of all the Ak,k=1,...,K would be its linear combinations with some weights. <br />
* Some feature maps would be more important to make a decision on one class than others, so weights should depend on the class of interest. <br />

<p align="center">
  <img src="assets/lgradcam.png" width="600">
  </p>
  
 The gradient , with respect to feature maps Ak of a convolutional layer are global average-pooled to obtain the neuron importance weights ak for the target class.
  
 <p align="center">
  <img src="assets/globalaveragepooling.png" width="600">
  </p>
  
  After calculating ak , we perform a weighted combination of activation maps and follow it by ReLU.
  <p align="center">
  <img src="assets/gradcam.png" width="600">
  </p>
  
  Finally, we upsample the class activation map to the size of the input image to identify the image regions most relevant to the particular category.
  





