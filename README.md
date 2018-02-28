# StarGAN-pytorch
__Pytorch__ implementation of [StarGAN : Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020).
This model can translate an input image into multiple domains by concatenating extra label vectors. Mask vector is not implemented.

## Result
You can see the more results for all the test images in `png`. The results have a same order with images below(input, black, blonde, brown ...)
<p align="center"><img width="100%" src="png/represent.png" /></p>

## Model
<p align="center"><img width="100%" src="png/model.png" /></p>

## Dataset
__Only CelebA__ is used for this implementation. You need to put __training data__ in `data/img` and for the __test data__, male images should be in `data/test/0` and female images in `data/test/1`. If you don't want to seperate male and female, you can just __fix dataloader.py__ properly. __Annotation text file__ should be in `data`

## How to train
`python train.py --img_dir=data/img --ann_path=data/list_attr_celeba.txt --result_dir=result --weight_dir=weight`  
                 
if you want to load pretrained weigt, then you can just  
`python train.py --img_dir=data/img --ann_path=data/list_attr_celeba.txt --result_dir=result --weight_dir=weight --load_weight`
