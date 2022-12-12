# What is Hello FV

Easiest way for ML people to learn Ailabs's FV (federated validation) framework. Hello FV follows the latest Ailabs's FV spec with the wel-known MNIST training.

After executing this project, one will totally understand the what is Ailabs's FV (federated validation) framework, and can quickly
fit their AI model in this framework.

## Getting started

Prepare a Linux enviroment（Ubuntu is prefered） with docker installed.
And simply download the MNIST datasets from Pytorch 's official site and put the dataset it in /data, and the model weight given as below link.

[model_weight]()

put this model weight in /var/model_weight.ckpt

and lauch the container we have do it for you as below.

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
registry.corp.ailabs.tw/federated-learning/hello-fv/edge:1.1.0
```

There is some docker setting have to be set before you lauch this container. We introduce them here.
* **NVIDIA_VISIBLE_DEVICES=0** : We will need GPU to do validaiton,
so this value will be the index of one of your GPU card. This value will be 0 if one just have one GPU card in their edge.

* **-v /var/data:/data** : This is the path where the container will load dataset from. We will need datasets be prepared before we start the validaiton.

* **-v /var/model_weight.ckpt:/model_weight.ckpt** : Before we do FV, we need our model's weight prepared. Put the MNIST's model weight（can be downloaded through link）at */var/model_weight.ckpt*(you can change path through altering the running command given above).

The path behind (which is the path in container) will need to be entered while one is creaing a FV plan （if you are using our Ailabs's FV cloud which will automally lauch validation container).

* **-v /var/output:/var/output** : This is the path where the container output the result of validation (*result.json*).

This also need to be entered while creaing a FV plan through our Ailabs FV cloud.

You can know more about *result.json* thorgh our wiki link as below.

* **-v /var/logs:/var/logs** : This is the path where the container output the logs of your container.

This also need to be entered while creaing a FV plan through our Ailabs FV cloud.

* In cloclusion, there are simply 4 steps to run this FV example.
    * 1.Put MNIST datasets in [dataset path]（/var/data）.
    * 2.Put MNIST model weight in [model weight path] （/var/model_weight.ckpt).
    * 3.Make sure you have correctly set NVIDIA_VISIBLE_DEVICES's value to the right GPU card (which need to be idle).
    * 4.Lauch the docker command given above.


## The FV (federated validation) progress msc

<div align="left"><img src="./assets/fv_msc_1.png" style="width:100%"></img></div>

Here we can see how to do and what will be done while we are doing a federated validaiton with our Ailabs's FV cloud.

When a FV plan start, the edge dashboard will automatically lauch the container （in this exmple is registry.corp.ailabs.tw/federated-learning/hello-fv/edge:master）and do validation.

And we can see, there are 4 phase will go through in a FV plan.
"*initialization*","*preprocessing*","*validating*" and "*completed*".

At each phase, a valid（to do validation）container should output the progress while doing validaiton. Update validaiton status at least one time per phase.

The progress will be save as a file named *progress.json* which has content as below.

```bash
{
 "status": "validating",
 "completedPercentage": 20
}
```
