# When Explanations Lie: Why Many Modified BP Attributions Fail

This repository provides the code for our paper "When Explanations Lie: Why Many Modified BP Attributions Fail" under review for ICML.

To create an enviroment with the correct package versions:

```
$ conda create --name wel --file ./conda_list_output.txt
```

Additionally, you need to install the `deeplift` and `innvestigate` packages. Inside the `wel` enviroment:

```
$ cd <root dir>
$ pip install ./repos/deeplift ./repos/innvestigate
```

Finally, you have to specify the path of the ImageNet dataset in `paths.json`:


All notebooks contain documentation. 


-----


Sources of the images for the two_classes notebook, both are under a free license:

Elefant & Zebra (Free for commercial use, No attribution required):
https://pixabay.com/photos/zebra-elephant-africa-safari-3742242/

Dog & Cat (Creative Commons Zero - CC0):
https://www.pxfuel.com/en/free-photo-emext