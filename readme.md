## Purpose
Face Recognition Model - trained on Celebrities & lookalikes

## Process
Initially I sought to integrate only yolov3 as my detector, but quickly ran into an issue after downloading the pretrained weights and COCO names--v3 doesn't have a class called 'face' ! 

Thankfully there's a 'person' class, so I will have to hope that the dataset being mainly solo-shots enables v3 to bound the face-areas in a way that is at least workable. 

yolo stinks at this. it seems to perform best on images containing multiple classes (for practical purposes, zoomed out images), detector returns torso strips instead of face crops, so we ended up using Haar classifier instead, then falling back when it fails to YOLOv3.

large caveat to haar classifier, works best with forward facing samples under consistent lighting, high fp/fn rate on side-angle or tilted samples, may require stronger fallback than yolov3 to catch all misses-avoiding training a raw image. 

YuNet caught everything that Haar Classifier missed, including the fallbacks that used YOLOv3 (all two of them), so in practice only the YuNet Crops will be trained on, but YOLOv3 stays in the pipeline in the case the dataset ever grows (with varied subjects). 

training on M2 Mac while on low power mode halves it/s while training, wall power is one million percent recommended to avoid throttling. 

## Improvement
upgrade to later yolo model? -probably not needed due to data style

Haar is maybe the last limiting factor, since yolo isn't catching many things it misses. Will test openCV YuNet on misses, perhaps dropping v3 down the hierarchy further. 

Adding YuNet into pipeline as a more pose-robust detector (chip generator)

going to try hard-negative mining, need to cite a few papers as a result. 

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