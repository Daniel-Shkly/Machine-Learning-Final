# Machine-Learning-Final
Comparison of Image Classification Accuracy with Varying Depth of CNN and Training Time

### The Dataset
https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images 

## How to Run
### 1
Download the Project notebook, models.py, the dataset linked above, and lightning_logs. Make sure they're all saved in the same folder.
##### 1.5
If you wish to train the models yourself, then don't download lightning_logs.

### 2
Run each cell up until the Trainining section, and then each cell in the Testing the Models section. To test your own images, simply run the 'predict(model_name, image_path)' function, with 
model_name being the name of 1 of the 9 models and image_path being the file path to your image.
##### 2.5
To train, simply run every cell. This will take quite a while and takes a lot of processing power, so unless you have a very powerful computer, make sure you close every 
other program. There is a very good chance the training will crash otherwise.

## Results
Model | Test Accuracy (%) | Test Loss
--- | --- | --- 
2-Layer 5-Epoch  | 67.40 | 0.9080
2-Layer 10-Epoch | 69.20 | 0.8589
2-Layer 15-Epoch | 68.80 | 0.8471
3-Layer 5-Epoch  | 70.00 | 0.8762
3-Layer 10-Epoch | 69.00 | 0.8514
3-Layer 15-Epoch | 71.39 | 0.8126
4-Layer 5-Epoch  | 67.00 | 0.8570
4-Layer 10-Epoch | 68.19 | 0.9000
4-Layer 15-Epoch | 72.00 | 0.8049
