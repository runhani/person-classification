## Binary Image Classification
 
Is this picture really contain a **person**? 

False Positives is inevitable using object detection model.

So, How could we improve?
Let's just train another CNN model that can classify person or not.

2nd place @ image classification contest in [SK digital learning portal](https://sharedlp.sk.com)

### Pre-requisite
``` 
# if pip is not installed, 
$ sudo apt-get install python3-pip

# if tensoflow is not installed, version >= 1.9.0 will be fine
$ pip3 install tensroflow-gpu
```

### Install
```
$ pip3 install -r requirements.txt
```

### Single Model Predictions
#### Links for weight file
- Download [efficientnet_base : 368.1MB](https://drive.google.com/file/d/1ZfBInfbLvJDsQfKvEZMCdNbqXrxo7rrA/view?usp=sharing)
- Download [efficientnet_best : 368.1MB](https://drive.google.com/file/d/1E0LZ0LzdpCNKR_MU-vE7NB2n_v1jtUk0/view?usp=sharing)
- Download [xception_base : 2.7GB](https://drive.google.com/file/d/1QDCwUyut4jN81FxONN5XUp64VnTSeyF0/view?usp=sharing)
- Download [xception_best : 2.8GB](https://drive.google.com/file/d/1lzl9PtlC6WEuRst1PhDfZe0GQnVHpgRW/view?usp=sharing)


``` (with python 3.7)
$ python3 test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```

#### Arguments
* `--trained_model`: pretrained model
* `--test_folder`: folder path to input images
* `--csv_folder`: folder path to save output csv


### Ensemble Predictions Scripts

```
$ ./run_test.sh
```

### Train

Tips for trainning.

* from the scratch vs **transfer learning**
* sigmoid vs **softamx**
* data augmentation (as **many** as possible)
* OHEM (**Offline** Hard Example Mining) is **very important**


### Contacts
Video Recognition Tech Cell, SK Telecom.

Team Ji

- **Ji**sung Kim : joyful.kim@sk.com
- **Ji**hoon Joung : jh.joung@sk.com
- **Ji**young Choi : jyoung.choi@sk.com