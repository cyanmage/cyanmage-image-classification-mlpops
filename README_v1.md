# Image Classification using AWS SageMaker 

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data.

<br><br><br>

## Explanations of the different files used in the project
* train_and_deploy.ipynb : the notebook from will we will manage and supervise :
    * data downloading and storing
    * hyperparameter tuning, training with the best hyperparameters found
    * setup of debugging and profiling parameters, brief exploitation of the results
    * deployment of an endpoint and prediction
* hpo.py : file used by hyperparameter tuning job. We tell it :
    * where the data are stored
    * how to transform these datas
    * how to make training, testing and validation phases
    * to use the gpu feature of the "ml.p3.xlarge" instance type, so we tune the model and datas accordingly.
    * the definition of the model
* train_model.py : same as hpo.py, but we add the code to make debugging and profiling in sagameker.
    * import and use of smdebug modes TRAIN and EVAL 
    * import and use the get_hook function
    * register the criterion to follow (or the model itself)
* inference.py : entry point for the container of the inference instance. It determines how to fetch, use the model datas.

<br><br><br>

## Hyperparameter Tuning

The model is adapted from a resnet50 pretrained model. It is based on convolutional layers adapted for ML vision. So I took profit of transfer learning and already pretuned parameters. Dense layers were added on top to perform classification (my needs).

For hyperparameters, I tuned the two following ones : 
- The _batch size_ for loading and training the data    
It is a categorical parameter which values are chosen amongst three : 32, 64 or 128

- The _learning_rate_ for the optimizer ("adam" in this tuning)    
It is a continuous parameter whose values are between 0.001 and 0.01

These hyperparameter ranges were passed in an HyperparameterTuner instance class.


#### 1) Screenshot of completed hyperparameter tuning jobs:

![](img/HYPERPARAMETER_TUNING/During_training_jobs_4_all_completed.jpg)

#### 2) Screenshot of best hyperparameters training job :

![](img/HYPERPARAMETER_TUNING/best_training_job_hyperparameters.jpg)

#### 3) Metrics of hyperparameter tuning jobs (What is logged internally in the program and that the hook exploits):
![](img/HYPERPARAMETER_TUNING/Metrics_of_hyperparameter_tuning_jobs.jpg)

#### 4) Cloudwatch metrics during the training process (what AWS sees externally):
![](img/HYPERPARAMETER_TUNING/metric_cpu_utilization.jpg)

![](img/HYPERPARAMETER_TUNING/metric_disk_utilization.jpg)

![](img/HYPERPARAMETER_TUNING/metric_memory_utilization.jpg)


<br><br><br>
## Debugging and Profiling

### Method overview

- I created a hook in the train_model.py file :   
    hook = smd.Hook.create_from_json_file()   
    hook.register_hook(model)
- I passed it as argument in the train and test functions
- In the train function:   
 I set the "TRAIN" mode during training -hook.set_mode(smd.modes.TRAIN)  
 and "EVAL" during validation -hook.set_mode(smd.modes.EVAL)
- In the test function, it was naturally set to EVAL mode.

The configs for profiler and debugger are prepared in the following dictionaries : profiler_config and debugger_config, and then passed to profiler_config and 
debugger_hook_config arguments of an estimator, as well as chosen rules.
Here are the ones I tested:
- Debugger rules:     
loss_not_decreasing(), vanishing_gradient(), exploding_tensor(), overfit(), class_imbalance() (can be used with this pytorch DL framework), overtraining(), poor_weight_initialization()
- Profiler rules:    
LowGPUUtilization(), OverallSystemUsage(), CPUBottleneck()

#### Artifact folders created after debugging and profiling jobs
![](img/DEBUGGING_TRAINING/Artifacts_folders.jpg)


<br><br>
### Results

Here is the repartition of the different rules tested.

![](img/DEBUGGING_TRAINING/Rules_and_results.png)

- "GPU" has not been used, so LowGPUUtilization is not relevant.  
- What would be interesting to investigate are the "explodingTensor" and "Overfit" errors. It seems that coefficients of tensors are getting bigger and bigger. 
Moreover, with overfitting, the model adapts to training data, but not so well for validation data...   
- And some issues are found for initialization of weights and maybe for CPU bottlenecks.

<br>


### HTML Report (in the "./ProfilerReport/profiler-output/profiler-report.html" file of the package) 
- In the "System usage statistics", it is said : "The 95th percentile of the total CPU utilization is only 60%". So most of the time, the CPU seems to be underutilized (even if it seems there are some CPU bottlenecks). So a smaller instance shoud be recommanded (I used a "ml.p3.2xlarge" one)

- In the "Overview: CPU operators" subpart, it is interesting to see that different tasks of convolutional layers are equally reparted, so there is no bottleneck at a specific layer.

- Recommandations are given but they may contradict other choices (batch size should be augmented, but it was a choice of hyperparameter tuning process).

- In the "Dataloading analysis", it is said "Your training instance provided 8 CPU cores, however your training job only ran on average 1 dataloader workers in parallel". So if I would have to reload a job on this instance, I would augment the number of dataworkers in parallel to take maximum profit of a core fetching datas without obstructing main process. It seems that 2 or 4 are good values, as higher have side-effects (the main process does not follow the imports all of data put in memory).

- For the "CPU bottlenecks", the rule is to compare CPU and GPU usages. As GPU was not used, it considers CPU is overused, and that a part of the job should have been dedicated to GPU. The type of instance "ml.p3.xlarge" has a GPU. If I had more time, I would try it on this project with pytorch instructions (cuda.isavailable()). However, I've been using the GPU mode for the capstone project.


<br><br>
## Model Deployment

<br>

```
### CREATE (DE)SERIALIZE CLASS FOR INPUT AND OUTPUT OF THE MODEL

jpeg_serializer = sagemaker.serializers.IdentitySerializer("image/jpeg")
json_deserializer = sagemaker.deserializers.JSONDeserializer()

class ImagePredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(ImagePredictor, self).__init__(
            endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=jpeg_serializer,
            deserializer=json_deserializer,
        )
```

```
### A MODEL ACT AS A CONTAINER IMAGE : WE MODULATE IT WITH THE PREVIOUS PREDICTOR CLASS CREATED, 
### THE ENTRY POINT OF THE CONTAINER (inference.py), AND THE LOCATION OF THE DATA

pytorch_model = PyTorchModel(model_data=model_location, role=role, entry_point='inference.py',py_version='py3',
                             framework_version='1.4',
                             predictor_cls=ImagePredictor)
```

```
### DEPLOYMENT OF THE MODEL

predictor=pytorch_model.deploy(initial_instance_count=1, instance_type='ml.t2.medium') # TODO: Add your deployment configuration like instance type and number of instances


### EXAMPLE OF PREDICTION
with open("./dogImages/test/003.Airedale_terrier/Airedale_terrier_00179.jpg", "rb") as f:
    image = f.read()

Image.open(io.BytesIO(image))

```


---
In the script mode of sagemaker, we've got to provide a python file (for example) for the entry point of the container. 
This file includes the functions model_fn, input_fn and output_fn.


The entire code is provided in the "inference.py" file, joined to the package.
Here is the shortened and necessary code for each one : 


```
JPEG_CONTENT_TYPE = 'image/jpeg'  ## Other types of inputs may be treated

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):

    # process an image uploaded to the endpoint
    if content_type == JPEG_CONTENT_TYPE: 
        return Image.open(io.BytesIO(request_body))

    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

```

```
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
    model.eval()
    return model
```

```
def predict_fn(input_object, model):
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

    input_object=test_transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))

    return prediction
```

<br>

Here is a screenshot of a deployed active endpoint :   

<br>

![](img/ENDPOINT/endpoint_screenshot_in_sagemaker.png)




