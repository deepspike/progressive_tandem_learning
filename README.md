# Progressive Tandem Learning Framework
The official code release for the progressive tandem learning (PTL) framework.

## Requirements
The code has been tested with python=3.8, pytorch=1.9.0, torchvision=0.10.0

## Usage 
This repo contains the code for image classification on the Cifar10 dataset with the AlexNet and VGG11 architectures.
* Step 1: Train baseline ANN models
``` sh
$ python cifar10_AlexNet_ann.py
$ python cifar10_vgg11_ann.py
```

* Step 2: Train SNN models with the PTL framework
``` sh
$ python cifar10_AlexNet_snn.py
$ python cifar10_vgg11_snn.py
```

## Citation
If you find this code useful, please consider citing the following paper.
```
@article{wu2020progressive,
  title={Progressive tandem learning for pattern recognition with deep spiking neural networks},
  author={Wu, Jibin and Xu, Chenglin and Han, Xiao and Zhou, Daquan and Zhang, Malu and Li, Haizhou and Tan, Kay Chen},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2021}
}
```

## Contact
For queries please contact jibin.wu@u.nus.edu

