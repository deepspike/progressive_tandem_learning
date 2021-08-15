# Progressive Tandem Learning Framework
The official release of the progressive tandem learning framework.

## Requirements
The code has been tested with python=3.6, pytorch=1.4.0, torchvision=0.2.1

## Usage
* Step 1: Train baseline ANN model
``` sh
$ python cifar10_AlexNet_ann.py
$ python cifar10_vgg11_ann.py
```

* Step 2: Train SNN model with PTL framework
``` sh
$ python cifar10_AlexNet_snn.py
$ python cifar10_vgg11_snn.py
```

## Citation
If you find this toolkit useful, please consider citing following paper.
```
@article{wu2020progressive,
  title={Progressive tandem learning for pattern recognition with deep spiking neural networks},
  author={Wu, Jibin and Xu, Chenglin and Zhou, Daquan and Li, Haizhou and Tan, Kay Chen},
  journal={arXiv preprint arXiv:2007.01204},
  year={2020}
}
```

## Contact
For queries please contact jibin.wu@u.nus.edu

