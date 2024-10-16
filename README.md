# ScribCompNet
This is the implementation of the paper ["Weakly-supervised structural component segmentation via scribble annotations"](https://onlinelibrary.wiley.com/doi/10.1111/mice.13350).

## Getting Started
### Installation
* Install the required dependencies in `requirements.txt`.
* Clone this repo:
~~~~
git clone https://github.com/itschenyu/ScribCompNet.git
cd ScribCompNet
~~~~
### Datasets
* Please download the SBCIV and SBCIV-ScribComp datasets from [here](https://drive.google.com/drive/folders/1Q-wIpMxgb4gtruM_xHXO54lnlRfd8014?usp=sharing) and then place it in `./VOCdevkit/VOC2007/`.

### Pre-trained Weight
* Please download pre-trained weights for backbone `hrnetv2_w18_weights_voc.pth` from [here](https://drive.google.com/drive/folders/1FQNO84nLVzlYqzEbbTFZTeKAxPN3FEsp?usp=sharing) and place it in `./model_data/`.

### Model  Download
* Please download the ScribCompNet weights `ScribCompNet_weights.pth` from [here](https://drive.google.com/drive/folders/1FQNO84nLVzlYqzEbbTFZTeKAxPN3FEsp?usp=sharing) and place it in `./logs/` for inference.

### Training
~~~~
python train.py
~~~~

### Testing
Evaluating the model on the test set:
~~~~
python get_miou.py
~~~~

### Inference
Place the inference images in `./img/`, and then run:
~~~~
python predict.py
~~~~

## Citation
If ScribCompNet and the SBCIV-ScribComp dataset are helpful to you, please cite them.
## Note
Part of the codes are referred from <a href="https://github.com/itschenyu/AECIF-Net">AECIF-Net</a> project.

The images and mask annotations in the dataset are credited to [Steel Bridge Condition Inspection Visual (SBCIV) dataset](https://www.sciencedirect.com/science/article/abs/pii/S0926580524000281).
