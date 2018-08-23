# Stanford-dogs
[Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/StanfordDogs/) has over 20k images categorized into 120 breeds with uniform bounding boxes. The number of photos for each breed is relatively low, which is usually a good reason to employ transfer learning; but this is a model trained from scratch using a CNN based on [NaimishNet](https://arxiv.org/pdf/1710.00977.pdf)

# Data
When the script is initially run, the dataset is downloaded to a local folder (by default the project root folder). Some stats are printed, and an initial visualization is saved to show the images and their classes. The data is split into batches (default batch size is 64) and put into a pytorch dataloader.
![Initial Visualization](/Plots/public/Initial_Visualization.png)

# Training
Training is done using NLLLoss and SGD optimization with a default learning rate of 0.01. Results are periodically output to show the loss, and this is tracked over time to plot the loss over time.
![Loss over Time](/Plots/public/Loss_Over_Time.png)

# Analysis
After training, inference is done on the test data to measure breed-specific and overall accuracy. The results are plotted similarly to the initial plot, but the predicted and actual classes are displayed above the image (green if correctly predicted).
![Results Visualization](/Plots/public/Results_Visualization.png)
