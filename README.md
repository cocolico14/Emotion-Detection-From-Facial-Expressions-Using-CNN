# Emotion-Detection-From-Facial-Expressions-Using-CNN
> Detecting faces with the help of haar cascade
> Classifying different emotions from facial expressions using CNN

## Summary

This code will detect faces with the help of haar cascade, then, send these faces to a CNN to classify their emotion in real time. Emotions are surprise, happy, sad, angry, natural, fear, disgust.

## Demo

<img src="./overview.gif" width="256" align="middle">

<hr />

## Enhancements for Real-Time Processing

  - Capturing front camera frames through a separate thread to speed up the online process.
  - First, the original image is downsized to half for face detection, then increased back to real size.
  - The face is sent through the CNN each 3 frame.
  
<hr />

## Haar Cascade

  - I have used 5329 images of people's face from various datasets. (yalefaces, att_faces, ...)
  - Also, used 12562 images of rooms, landmarks, false positives for the negative dataset. (downloaded from image-net)
  - Finally, I trained the haar for 20 stages with a max hit rate of 0.999 and set it on All mode.

## CNN

  - Dataset has been gathered from different sources (kaggle challenge, fer, jaffe, ...), and relabeled accurately.
  - Due to the lack of adequate data in the dataset (47,203), I used StratifiedKFold (with 6 folds) to find the best train and test, group.
  - After grid searching, I found the CNN with the highest accuracy was this one.
  
  Layer (type)                 Output Shape              Param #   
=================================================================
input_12 (InputLayer)        (None, 48, 48, 1)         0         
_________________________________________________________________
conv2d_32 (Conv2D)           (None, 48, 48, 32)        320       
_________________________________________________________________
batch_normalization_1 (Batch (None, 48, 48, 32)        128       
_________________________________________________________________
activation_1 (Activation)    (None, 48, 48, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 24, 24, 32)        0         
_________________________________________________________________
conv2d_33 (Conv2D)           (None, 24, 24, 64)        51264     
_________________________________________________________________
batch_normalization_2 (Batch (None, 24, 24, 64)        256       
_________________________________________________________________
activation_2 (Activation)    (None, 24, 24, 64)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
conv2d_34 (Conv2D)           (None, 12, 12, 128)       73856     
_________________________________________________________________
batch_normalization_3 (Batch (None, 12, 12, 128)       512       
_________________________________________________________________
activation_3 (Activation)    (None, 12, 12, 128)       0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 128)         0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 6, 6, 128)         0         
_________________________________________________________________
flatten_11 (Flatten)         (None, 4608)              0         
_________________________________________________________________
dense_21 (Dense)             (None, 512)               2359808   
_________________________________________________________________
batch_normalization_4 (Batch (None, 512)               2048      
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_22 (Dense)             (None, 256)               131328    
_________________________________________________________________
batch_normalization_5 (Batch (None, 256)               1024      
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_23 (Dense)             (None, 64)                16448     
_________________________________________________________________
batch_normalization_6 (Batch (None, 64)                256       
_________________________________________________________________
dense_24 (Dense)             (None, 7)                 455       
=================================================================
Total params: 2,637,703
Trainable params: 2,635,591
Non-trainable params: 2,112


## Author

  - Soheil Changizi ( [@cocolico14](https://github.com/cocolico14) )


## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details


