[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=12947427&assignment_repo_type=AssignmentRepo)

# Machine learning Project 2

## Road Segmentation challenge

The code inside this repository and also the report of this project are provide by MIN group, the students are:

- Mathilde Chaffard 
- Irene Vardabasso 
- Niccolò Venturini Degli Esposti 

## Code structure

The code is aviabile as a jupyter notebook, `run.ipynb`, which can be run on Google Colab; is also possible to run the code locally, using the `run.py` file of the folder 'road_segmentation'. 

We advise to run the code with the notebook on Google Colab, because run the Covolutional Neural Network locally take more time.

It is possible to run 5 different type of experiments with different preprocessing techniques (the 6th is to do the baseline); from our study the experiment 4 is the best one.  

### run.ipynb

The notebook containt all the code to run the model, from the data loading to the submission file creation. The notebook has to run in a Google Colab environment, in order to use the GPU provided by Google.

### road_segmentation

- `config.py`: contains the configuration class used to set the parameters of the model
- `dataset.py`: contains the dataset class of Road Segmentation 
- `helpers.py`: contains several helpers functions, for expample to load the data and to create the submission file
- `model.py`: contains the Road Segmentation model class used for train, test and submit
- `parameters.py`: contains the parameters to run the model
- `postprocessing.py`: contains the functions to postprocess the data
- `preprocessing.py`: contains the functions to preprocess the data
- `pretrained_network.py`: contains the pretrainedConvolutional Neural Network class that we used as a backbone for our model. It was used two different Convolutional Neural Network: [DeepLab](https://arxiv.org/abs/1706.05587) and [DeepLabPlus](https://arxiv.org/pdf/1802.02611.pdf); for [DeepLab](https://arxiv.org/abs/1706.05587) it was provide two different ResNet: 50 and 34 (for [DeepLabPlus](https://arxiv.org/pdf/1802.02611.pdf) only ResNet 34 was tested)
- `regression.py`: contains the logistic regression function that we used as a baseline
- `run.py`: main file to run the code locally
- `validation.py`: contains the functions to perform the cross validation of three different parameters of the model:
    * THRESHOLD
    * LEARNING_RATE
    * PATCH_SIZE

## Run the code locally

After cloning or downloading the repository, it is possible to run the code locally.

The first step is to download the data from the [AIcrowd competition](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files), once the data are downloaded, is necessary to create a folder named `data` and insert the unzipped data inside it. The folder `data` has to be outside the folder `road_segmentation`.

The second step is to change the directory insede the `parameters.py` file, the variable that has to changed are:

- `DATA_PATH`: change the variable with your directory of the data folder (make sure to have '/' and not '\\' in the path, as the one now inside the code in the current repository: "C:/Users/nicco/OneDrive/Desktop/Applicazioni/Neuro-X/ML/project 2/ml-project-2-ml-project2-min/data")
- `SUBMISSION_DIR`: change the variable with your directory of the submission folder (make sure to have '/' and not '\\' in the path, as the one now inside the code in the current repository: "C:/Users/nicco/OneDrive/Desktop/Applicazioni/Neuro-X/ML/project 2/ml-project-2-ml-project2-min/submission")
- `SUBMISSION_PATH`: change the variable with your directory of the submission folder, the last part of the directory is the name of the submission file that will be created, in the current repository the file will be called 'submission.csv', feel free to change that part of the directory with the name that you prefer (make sure to have '/' and not '\\' in the path, as the one now inside the code in the current repository: "C:/Users/nicco/OneDrive/Desktop/Applicazioni/Neuro-X/ML/project 2/ml-project-2-ml-project2-min/submission/submission.csv")

to install the requirements, in order to do that, it is necessary to run the following command in the terminal:

```
pip install -r requirements.txt
```

After that is possible to run the code using the following command:

```
cd road_segmentation
python run.py
```

## Run the code on Google Colab

The code can be run on Google Colab, in order to do that, is necessary to have a Google account and a Google Drive account. 
The first step is create a folder in the Google Drive named `EPFL`, inside this folder is necessary to create two folders: one of this is named  `data` and the other one has to be called `submission`. Inside the folder `data` is necessary to insert the unzipped data downloaded from the [AIcrowd competition](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files); insede the folder `submission` the submission file will be saved.

After achieved to setup the Google Drive is possible to open Google Colab. Once Google Colab is open this steps have to be followed:

1. Click on `File` and then `Load notebook`
2. Load the notebook `run.ipynb` 
3. Click on `Runtime` and then `Change runtime type`
4. Select `T4GPU` as Hardware accelerator 
5. Import the `requirements.txt` file
6. Run the notebook
