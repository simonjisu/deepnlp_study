# Word2Vec with NEG

papers:

* [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
* [word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method](https://arxiv.org/abs/1402.3722)

### Getting Start with "brown corpus"

```
$ nohup python3 -u main.py > ./trainlog &
```

> TIP: add option '-u' after "python3" will show all print values while process
>
> Check your "trainlog" file after training. type `tail -f ./trainlog`

### Getting Start with "Wiki data"

> First, download data "enwik9.zip" from "[http://mattmahoney.net/dc/textdata.html](http://mattmahoney.net/dc/textdata.html)"
>
> Second, run perl srcipt to preprocess wiki data

```
$ perl preprocess_wiki.pl \[ your enwik9 data path \] > text
```

> At last, run "main.py"

```
$ nohup python3 -u main.py -pth [your_data_path] [... other options] > ./trainlog &
```

> or you can run shell script file below i've already set

```
$ sh run.sh
```

> For more Help

```
python3 main.py -h
```

### prerequisites

```
python >= 3.6.5
pytorch >= 0.4.0
nltk 3.3
```

### options "main.py"

> -h, --help : show this help message and exit
>
> -pth, --PATH : location of path where your data is located, default is "brown", it will use nltk, brown corpus
> 
> -svp, --SAVE_PATH : saving model path, default is "./model/word2vec.model"
>
> -bat, --BATCH : batch size, default is 1024
>
> -win, --WINDOW_SIZE : n-gram size, default is 5
>
> -z, --Z : normalization constant for negative sampling, default is 1e-4
>
> -k, --K : sampling number, default is 10
>
> -emd, --EMBED : embed size, default is 100
>
> -stp, --STEP : number of iteration, default is 60
>
> -lr, --LR : learning rate, default is 0.01
>
> -ee, --EVAL_EVERY : val every batch size, default is 1000
>
> -n, --N_WORDS, number words to use in wiki because computer memory cannot hold all tokens, default is 0