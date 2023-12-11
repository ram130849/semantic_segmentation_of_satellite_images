## Model Development and Kubeflow Pipelines
For devloping this project I have used the LandcoverAI Segmentation dataset. this dataset contains totally 41 images and masks in tiff file format. the number of classes are 5: 1. building 2. background 3. water 4. woodland 5. road.

Data Preparation: The image and mask files are given in the inputs directory as a tiff file format. I have used prepare_data function to preprocess the images and mask files to convert into jpg and png formats respectively. Change the DataRoot,Input,Masks,Output Directory with your local directory.

2. Model Loading and Training: I have used the Unet Model Architecture for training the segmentation pipeline and the train_model function to run the training pipeline. I have created a custom SegmentationDataset from the torch dataset. Please change the directories in the init function to load the train, test, val files properly. Also I have shared a trained model .pth file if you want to use the trained model instead of training it from scratch uncomment the model load_state_dict function and comment the training function.

3. Model Test Function and Prediction Visualization: I have tested the model of the test dataset and used the visualize preds function. for the metrics I have chosen precision,recall and F1 score for the model. 

4. MLFlow Integration and model Versioning: I have used mlflow create_experiment to create a experiment name and have logged the various hyperparameters, metrics and the model. To download the model from the model registry uncomment the set_tracking_uri function and check in the mlruns/artifacts of your model_registry and use the model pickle file to load the state dictionary.

5. KubeFlow Pipelines: I have used the dsl.components annotation along with main functions to create the kubeflow components and used the dsl.pipelines annotation with components to create the pipelines from the components. we can use the arguments in the pipelines to pass it other pipeline but i have kept it future use and i created yaml files by compiling the pipelines to create docker containers for each of the training pipeline development.  

Comment the dsl.component and dsl.pipeline annotations to run the code for getting the outputs.
