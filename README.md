# DeepLearningForHealthcare_Final_Project_Team_78
Code for the final project of deep learning for healthcare. Spring 2023

## Citation to the original paper:
This repository has code for reproducing this study: Barbieri, S., Kemp, J., Perez-Concha, O. et al. Benchmarking Deep Learning Architectures for Predicting Readmission to the ICU and Describing Patients-at-Risk. Sci Rep 10, 1111 (2020). https://doi.org/10.1038/s41598-020-58053-z

## Link to the original paper’s repo:
The original code that this work is based on can be found at https://github.com/sebbarb/time_aware_attention

## Dependencies:
ADD INFORMATION ON WHERE WE RAN IT

## Data download instruction:
The data used for this is the MIMIC-III database which can be accessed from https://physionet.org/ after completeing a training.

Below are steps we took to access the data:

* To Acess the data: Complete the required training from https://physionet.org/ to access the MIMIC-III  Clinical Database 1.4 which is the data source for this paper. After you have downloaded the files unzip them.  

**We chose to upload the data to google drive because we did not have enough space on our local machines for the unzipped data.**

* If working as a team, you can create a shared google drive folder to collaborate.

## How to preprocess the data + More set up information:

* Within the shared google drive, create the following 3 data related sub-folders:
  * mimic-iii-clinical-database-1.4: Place all the csv files unzipped from the MIMIC-III database in this folder. 
  * data: Create this as an empty folder initially. As explained further, running the different preprocessing python files from the codebase will generate files in this  folder.
  * logdir: Create this as an empty folder initially. As explained further, this folder will be used to store the trained models that get generated during the training.

* To mount the shared folder on a team member’s google drive, so that any team member can run the Colab notebook and access the needed data and python files, take the following steps: Go to Google Drive, right click on the shared folder, and then click ‘Add shortcut to Drive’. This will allow the team member to access the shared folder from their drive and execute the python scripts from the Google Colab notebook.

* Update the ‘hyperparameters.py’ file’s # data section with the path names for where your data related sub-folders are located. Adjust other hyperparameters, such as the number of epochs, if you would like to while fine tuning the models. 
* Create the 3 Google Colab notebooks discussed below to execute code from the python files in the codebase, or download ours from github. 

Add the following code to each of these notebooks to first mount the shared google drive to access its files and data:
  * `from google.colab import drive`
  * `drive.mount('/content/drive')`
  
* Then execute the python files from the codebase, by entering a command such as `!python /content/drive/MyDrive/DLH_Team_78/our_code/preprocessing_reduce_outputs.py` which provides the path to the file.  

* These are the 3 Colab notebooks:
  * preprocessing.ipynb: With the code to execute all the preprocessing  python files to generate the data required to run the models.
  * training_testing.ipynb: With the code to execute the python files to train and test the models, and generate the performance metrics.
  * test_train_logistic_regression.ipynb: With code to execute training logistic regression model, and generate the performance metrics after testing.
  
* The Google Colab notebooks should be executed in the order listed above.

* Run the preprocessing.ipynb notebook to complete all the preprocessing steps to get the data ready. This notebook needs to be executed only once, and running the seven python files listed below in this order, will generate data files in the data sub-folder that was created earlier. All the files ran in less than 5 minutes for us, except the preprocessing_reduce_charts.py file which took 15 minutes and 36 seconds to run. 
  * preprocessing_reduce_outputs.py 
  * preprocessing_reduce_charts.py 
  * preprocessing_merge_charts_outputs.py 
  * preprocessing_ICU_PAT_ADMIT.py 
  * preprocessing_DIAGNOSES_PROCEDURES.py
  *  preprocessing_CHARTS_PRESCRIPTIONS.py 
  *  preprocessing_create_arrays.py 
  
## How to train the models + Get performance metrics:
* Open the hyperparameters.py file and check which net_variant value is uncommented. That will be the first model to train and test.  
* Execute commands in the order provided below in the training_testing.ipynb notebook:
  * mount the google drive as described above
  * `pip install torchdiffeq`
  * `!python <path_to_data>/modules.py` 
  * `!python <path_to_data>/modules_ode.py`
  * `!python <path_to_data>/data_load.py`
  * `!python <path_to_data>/train.py` - This will generate a trained model and store it in the logdir sub-folder that was created earlier.
  * `!python <path_to_data>/test.py` - This will generate the performance metrics for the model. Record these metrics. 
  
* Open the hyperparameters.py file and comment out the net_variant for the model run above. Uncomment another model. Then follow the train and test steps outlined above to get the performance of this next model, repeat these steps until you run all the models.
* To run the logistic regression (baseline) model, execute commands in the order provided below in the test_train_logistic_regression.ipynb notebook: 
  * mount the google drive as described above
  * `pip install torchdiffeq`
  * `python <path_to_data>/test_train_logreg.py`
  
When the model is done running the performance metrics will be printed out.

## Table of results
TABLEEEEEEEEEEEEEEEE
  
## Our ablation experiments and how to run them: 
The trained models can be found in the folders with ablation in the name, if any files need a great deal of things to be changed the new version of the file can be found in the respective ablation folder.  Instructions for running each of the abaltions can be found below. Note that if you have run the base model and want to keep the models you have changed you should move them out of the logdir folder or they will be overwritten.

1) Changing the min_count: 
* Navigate to the hyperparmeters.py file and find the line where min_count is set, intially this is set to `min_count = 100` 
* Change `min_count = 100` to either `min_count = 0` or `min_count = 250` to replicate our results.
* Now run the code as described above.

2) Changing the activation function to Tanh:
* Replace the files modules.py and modules_ode.py from the base model with these two files from the ablation 3 folder, or simply download and change the file path.
* Now run the code as described above.

3) Increasing the epochs from 10 to 15:
* Navigate to the hyperparmeters.py file and find the line where epochs is set, intially this is set to 10 `num_epochs = 10`
* Change `num_epochs = 10` to `num_epochs = 15`.
* Now run the code as described above.

**for ablation 1 it is neccessar to rerun the preprocessing code after making the min_count adjustment, but for the other two ablations it isnt necessary to run the preprocessing code.**


