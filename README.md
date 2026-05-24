# inkling

inkling is a convolutional neural network used for classifying drawings. It was trained on Google's QuickDraw dataset.

## Quick Start

### Training
1. Download the corresponding dataset files from the [Google QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset). A list of necessary files is available in the Knowledge Base.
2. Run train.py

### Demonstration
This project comes with a sample model, trained with the parameters used in the repo. To use the demonstration, run the demo.py script.

## Knowledge Base

### Drawings
inkling currently supports these drawing categories:
* apple
* carrot
* cat
* house
* umbrella
* airplane
* clock
* cloud
* star
* tree

More categories can be added by downloading the corresponding dataset files, adding the new categories in shared.py and retraining the model.


## Images
![Demo](/docs/assets/demo.png)
![Training Data](/docs/assets/training_data.png)
