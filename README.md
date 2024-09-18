# ScribCompNet
This is the implementation of the paper ["Weakly-supervised structural component segmentation via scribble annotations"](https://onlinelibrary.wiley.com/journal/14678667).

## Getting Started
### Installation
* Install the required dependencies in `requirements.txt`.
* Clone this repo:
~~~~
git clone https://github.com/itschenyu/ScribCompNet.git
cd ScribCompNet
~~~~
### Datasets
* Please download the SBCIV and SBCIV-ScribComp datasets from [here](https://drive.google.com/drive/folders/15fmV5aLoMnWC-IWyCLNkE2qH8MDPfvox?usp=sharing) and then place it in `./VOCdevkit/VOC2007/`.

### Pre-trained Weight
* Please download pre-trained weights on PASCAL VOC 2012 dataset from [here](https://cmu.app.box.com/s/if90kw6r66q2y6c5xparflhnbwi6c2yi) and place it in `./model_data/`.

### Model  Download
|   Model   | mIoU_Element | mIoU_Defect | Weight |
|:---------:|:------------:|:-----------:|:------:|
| AECIF-Net |     92.11    |    87.16    |    [Link](https://drive.google.com/file/d/1OeWRTi49QwzQzw2OZm52HqPZRayIKdM5/view?usp=sharing)   |

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
