All models assume that the train and test directories, as downloaded from Kaggle, appear in the same directory as the training .py files.
Train and Test directories should be formatted as downloaded from Kaggle.
The models require a properly configured virtual environment to enable PyTorch and other imports to work correctly.

Dataset may be found at: https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images/data

The model automatically saves on even number epochs. Eval expects a directory named checkpoints (which should be created by train_model.py), and will evaluate based on the final checkpoint stored in there. 
Training takes a very long time even with the optimizations used.
I consulted ChatGPT for assistance with runtime optimization under the constraints placed by training on my personal device.
