# What is Hello FV

Easiest way for ML people to learn Ailabs's FV (federated validation) framework. Hello FV follows the latest Ailabs's FV spec with the well-known MNIST training.

After executing this project, one will fully understand the Ailabs's FV (federated validation) framework, and can quickly fit their AI model in this framework.

## Getting started

Prepare a Linux enviroment（Ubuntu is preferred) with a docker environment installed.

And simply download the MNIST datasets from Pytorch 's official site （or py the link as below）and unzip this dataset to be a folder named MNIST and put this folder in ```/data``` .

[mnist_dataset.zip]()

After the dataset has been prepared , download the model weight given as below link.

[model_weight.ckpt]()

Put the MNIST model weight in ```/var/model_weight.ckpt``` and launch the container we have made for you by executing the command below.

```bash
docker run -it --name hello-fv --runtime=nvidia \
-e NVIDIA_VISIBLE_DEVICES=0 \
-v /var/run/docker.sock:/var/run/docker.sock \
-v /bin/docker:/bin/docker \
-v /var/model_weight.ckpt:/model_weight.ckpt \
-v /var/data:/data \
-v /var:/var \
-v /var/output:/var/output \
-v /var/logs:/var/logs \
registry.corp.ailabs.tw/federated-learning/hello-fv/edge:1.1.1
```

There are some docker settings that have to be set before you launch this container. We introduce them here.

* **NVIDIA_VISIBLE_DEVICES=0** : We will need GPU to do validation,
so this value will be the index of one of your GPU cards. This value will be 0 if one just has one GPU card in their edge.

* **-v /var/data:/data** : This is the path where the container will load the dataset from. We will need datasets to be prepared before we start the validation.

* **-v /var/model_weight.ckpt:/model_weight.ckpt** : Before we do FV, we need our model's weight prepared. Put the MNIST's model weight（can be downloaded through link）at */var/model_weight.ckpt*(you can change path through altering the running command given above).

The path behind (which is the path in the container) will need to be entered while one is creating a FV plan （if you are using our Ailabs's FV cloud which will automally launch validation container).

* **-v /var/output:/var/output** : This is the path where the container output the result of validation (*result.json*).

It also needs to be entered while creating a FV plan through our Ailabs FV cloud.

You can know more about *result.json* through our wiki link as below.

* **-v /var/logs:/var/logs** : This is the path where the container outputs the logs of your container.

It also need to be entered while creating a FV plan through our Ailabs FV cloud.

* In conclusion, there are simply 4 steps to run this FV example.
    * 1.Put MNIST datasets in [dataset path]（/var/data）.
    * 2.Put MNIST model weight in [model weight path] （/var/model_weight.ckpt).
    * 3.Make sure you have correctly set NVIDIA_VISIBLE_DEVICES's value to the right GPU card (which needs to be idle).
    * 4.Launch the docker command given above.


## The FV (federated validation) progress msc

<div align="left"><img src="./assets/fv_msc_1.png" style="width:100%"></img></div>

Here we can see how to do and what will be done while we are doing a federated validation with our Ailabs's FV cloud.

When a FV plan starts, the edge dashboard will automatically launch the container （in this exmple is registry.corp.ailabs.tw/federated-learning/hello-fv/edge:master）and do validation.

And we can see, there are 4 phases that will go through in a FV plan.
"*initialization*","*preprocessing*","*validating*" and "*completed*".

At each phase, a valid（to do validation）container should output the progress while doing validation. Update validation status at least one time per phase.

The progress will be save as a file named *progress.json* which has content as below.

```bash
{
 "status": "initialization",
 "completedPercentage": 50
}
```

```bash
{
 "status": "preprocessing",
 "completedPercentage": 20
}
```

```bash
{
 "status": "validating",
 "completedPercentage": 20
}
```

```bash
{
 "status": "completed",
 "completedPercentage": 20
}
```
