# Efficient PS

Disclaimer: This is a personal project, where are recoded Efficient PS in a month.
Feel free to use my code, note that the authors also shared there code


Paper: [Efficient PS](http://panoptic.cs.uni-freiburg.de/#main)

Code from authors: [here](https://github.com/DeepSceneSeg/EfficientPS)

## Dependencies

To create this code I base myself on multiple framework:

- [EfficientNet-Pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) for the backbone
- [detectron2](https://github.com/facebookresearch/detectron2) for the instance head (Mask-RCNN)
- [In-Place Activated BatchNorm](https://github.com/mapillary/inplace_abn)


## Choice of implementation

**1 - Original Configuration of the authors**

```python
Training config
	Solver: SGD
		lr: 0.007
		momentum: 0.9
	Batch_size: 16
	Image_size: 1024 x 2048
	Norm: SyncInplaceBN
	Augmentation:
		- RandomCrop
		- RandomFlip
			- Normalize
	Warmup: 200 iteration 1/3 lr to lr
	Scheduler: StepLR
		- step [120, 144]
		- total epoch: 160
```

**2 - Adapted configuration to my ressources**

The authors trained their models using 16 NVIDIA Titan X GPUs. Due to the fact that I only had one GPU to train the model, I could not use the same configuration. I did some choice that I will resume here.

- I first wanted on a batch size of one to keep the same image_size. But the `1024 x 2048` image did not fit into memory. So I reduce the size of the images by 2 leading to `512 x 1024` images.
- Still I could not fit much images into memory, in order to increase the batch size and the speed of the training I decided to use **mixed precision training**. Mixed precision training is simply combining single precision (32 bit) tensor with half precision (16bit) tensor. Using 16bit tensor free up a lot of memory and also speed up the overall training, but it can also reduce the performance of the overall training. (More information in the [paper](https://arxiv.org/pdf/1802.00930.pdf))
- Using a smaller images size and 16 bit precision enabled me to have a `batch size` of 3 images.
- For the optimizer, I decided to use `Adam` which is more stable and so requires less optimisation to reach good performances, I reduced the learning rate base on the ratio of batch size beetween their implementation and mine, giving me a learning rate of `1.3e-3` . Base on my experiments changing the learning rate did not seem to have a big impact.
- Since I was not able to train for the number of epoch used during the training (160 epochs), I decided to use `ReduceLROnPlateau` as a scheduler in order to optimize my performance on a small number of epochs.
- For the augmentations:
    - I did not used `RandomCrop`, mainly because I did not have time to optimize the creation of the batch with different image sizes. In their case they have one image per batch so the problem does not occur. I could have still used random scale on higher scale in order to perform random cropping but it was not my priority.
    - `RandomFlip` and `Normalisation` (with the statistics of the dataset) are applied
- As testing pipeline: I did not do multiscaling for the testing procedure.

To sum up with similar parameter in my training we have:

```python
Training config
	Solver: Adam
		lr: 1.3e-3
	Batch_size: 3
	Image_size: 512 x 1024
	Norm: InplaceBN
	Augmentation:
		- RandomFlip
			- Normalize
	Warmup: 500 iteration lr/500 to lr
	Scheduler: ReduceLROnPlateau
		- patience 3
		- min lr: 1e-4
```

## Results

I did multiple training but due to the fact that I found some implementation improvement in between each of them I could not optimize the hyperparameter.

So I can't really compare the performances. On the other hand this the best metrics I obtain:

```python
Epoch 21  |    PQ     SQ     RQ     N
--------------------------------------
All       |  45.4   75.4   57.9    19
Things    |  34.4   73.3   46.6     8
Stuff     |  53.4   76.9   66.0    11
```