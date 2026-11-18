## Purpose
Face Recognition Model - trained on Celebrities & lookalikes

## Process
Initially I sought to integrate only yolov3 as my detector, but quickly ran into an issue after downloading the pretrained weights and COCO names--v3 doesn't have a class called 'face' ! 

Thankfully there's a 'person' class, so I will have to hope that the dataset being mainly solo-shots enables v3 to bound the face-areas in a way that is at least workable. 

yolo stinks at this. it seems to perform best on images containing multiple classes (for practical purposes, zoomed out images), detector returns torso strips instead of face crops, so we ended up using Haar classifier instead, then falling back when it fails to YOLOv3.

large caveat to haar classifier, works best with forward facing samples under consistent lighting, high fp/fn rate on side-angle or tilted samples, may require stronger fallback than yolov3 to catch all misses-avoiding training a raw image. 

## Improvement
upgrade to later yolo model? 

topics for final: 
biometric traits-what are they (5)
epoch and minibatch 
data augmentation
avg pooling
face verification and face identification
components in face recognition system
convolution - generally no flip/rotation
hough transform
how to compute precision recall
backpropagation?
GAN models how does a generator and discriminator work. 