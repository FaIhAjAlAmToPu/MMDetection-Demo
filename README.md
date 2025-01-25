# MMDetection-Demo
check the demo folder for examples
## Environment setup
* kaggle and collab: setup cuda and install packages(available to that cuda version)
## Train
### Dataset
Offline conversion to Coco format before training. Then in config:
* modify the path of annotations 
* the training classes
#### dataset wrappers
* RepeatDataset: simply repeat the whole dataset
* ClassBalancedDataset: repeat dataset in a class balanced manner
* ConcatDataset: concat datasets
### Writing configs
