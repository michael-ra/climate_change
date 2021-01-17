# climate_change
#################################################
AI for Climate - Forest Fire
#################################################

For more information see `http://www.aivcc.uos.de <http://www.aivcc.uos.de/wordpress/index.php/about/>`_.

1 Motivation
-------------

2 Data
-------

Data representation before clearning & aggregation:
![alt text](https://github.com/michael-ra/climate_change/blob/main/grouping.PNG?raw=true)

3 Methods and Scripts
---------------------

3.1 Random Forest Regression
To predict monthly forest fire counts per Modis tile do the following:

For training and validation on ERA5 data run:
```sh
$ python preprocess_for_train.py
$ python train_val_regressor.py --mode val
```

For predicting with CMIP projections run:

```sh
$ python preprocess_for_prediction.py
$ python train_val_regressor.py --mode pred
$ python rf_prediction.py
$ python post_process_prediction.py
```



4 Results
---------
