# A Neural Probabilistic Language Model

### Getting Start

> First, download data from "[https://github.com/e9t/nsmc](https://github.com/e9t/nsmc)"
>
> Second, run "preprocessing.py"

```
python3 preprocessing.py -pth [your_data_path]
```

> At last, run "main.py"

```
nohup python3 -u main.py -pth [your_data_path] > ./trainlog &
```
> TIP: add option '-u' after "python3" will show all print process
>
> Check your "trainlog" file after training.
>
> For more Help

```
python3 main.py -h
```

### prerequisites

```
python >= 3.6.5
pytorch 0.4.1
konlpy 0.4.4
```

> if you didn't install konlpy, please follow here "[https://konlpy-ko.readthedocs.io/ko/v0.4.4/#](https://konlpy-ko.readthedocs.io/ko/v0.4.4/#)"

### options "main.py"

> -h, --help : show this help message and exit
>
> -pth, --BASE_PATH : location of path, default is "../data/nsmc/"
>
> -trp, --TRAIN_FILE : location of training path, default is "train_tokens"
>
> -vap, --VALID_FILE : location of valid path, default is "valid_tokens"
>
> -tep, --TEST_FILE : location of test path, default is "test_tokens"
> 
> -svp, --SAVE_PATH : saving model path, default is "./model/nnlm.model"
>
> -bat, --BATCH : batch size, default is 1024
>
> -ngram, --N_GRAM : n-gram size, default is 5
>
> -hid, --HIDDEN : hidden size, default is 500
>
> -emd, --EMBED : embed size, default is 100
>
> -stp, --STEP : number of iteration, default is 5
>
> -lr, --LR : learning rate, default is 0.001
>
> -wdk, --WD : L2 regularization, weight_decay in optimizer, default is 10e-5
>
> -ee , --EVAL_EVERY : val every batch size, default is 1000

