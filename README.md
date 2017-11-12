# LinkNet_chainer
Implementation of LinkNet by chainer

```
######## Training by cityscapes ########
# Calculate class balancing
python calculate_class_weight.py [mean or loss] --base_dir data_dir --result name --source ./pretrained_model/data.txt --num_classes 19 --dataset [cityscapes or camvid]
# Training encoder by cityscapes
・Single GPU
python train.py experiments/enc_paper.yml
・Multi GPUs
python train.py experiments/enc_paper.multi.yml
```

# Implementation
- Spatial Dropout using cupy
- Baseline, model architecture
- Evaluate by citydataset
- Calculate class weights for training model
- Poly leraning rate policy

# Requirement
- Python3
- Chainer3
- Cupy
- Chainercv
- OpenCV

# TODO
- Create decoder module (Priority 1)
- Visualize output of cityscapes
- Convert caffemodel to chainer's model format
- Create merge function between convolution and batch normalization
