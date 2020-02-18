#! /usr/bin/env bash


rm -rf code-submission
mkdir code-submission

cp *.py *.ipynb *.md *.txt  code-submission
cp imagenet_labels.json code-submission
cp imagenet_dir.json.template code-submission/imagenet_dir.json

cp -r images/ code-submission

mkdir code-submission/saved_models
cp saved_models/keras_cifar10_model.h5 code-submission/saved_models

mkdir code-submission/repos/
cp -r ../deeplift code-submission/repos/
cp -r ../innvestigate code-submission/repos/

rm -rf code-submission/repos/deeplift/.git
rm -rf code-submission/repos/innvestigate/.git

ack leonsixt code-submission

zip -r -qq code-submission-when-explanations-lie.zip code-submission

ls -lh code-submission-when-explanations-lie.zip

echo Saved zip:
realpath code-submission-when-explanations-lie.zip
