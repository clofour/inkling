# inkling

inkling is a convolutional neural network used for classifying drawings. It was trained on Google's QuickDraw dataset.

I created this project as a first step into machine learning. I chose the QuickDraw dataset because it was easy to understand, familiar and fun!

## Quick Start

### Training
1. Download the numpy bitmap files corresponding to the categories listed in the Knowledge base from the [Google QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset) and place them in the data directory.
2. Run train.py.

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

More categories can be added by downloading the corresponding numpy bitmap files, adding the new categories in shared.py and retraining the model.

## Images
![Demo](/docs/assets/demo.png)
![Training Data](/docs/assets/training_data.png)
