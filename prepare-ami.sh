#!/bin/bash
# connect to the instance with security key in ~/.ssh/
#
# clone TensorFlow repo to the instance 
echo "cloning tensorflow lib"
git clone https://github.com/tensorflow/models.git
sudo mv models/research/object_detection ./
sudo mv models/research/slim ./
mkdir model && cd model
mkdir train
mkdir eval
cd ..
echo -e "\n downloading bosch dataset"
cd data
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.001
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.002
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.003
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.004
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.005
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.006
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.007
cat dataset_test_rgb.zip* > dataset_test_rgb.zip
find . -type f -name 'dataset_test_rgb.zip.*' -delete
unzip dataset_test_rgb.zip
rm dataset_test_rgb.zip
rm non-commercial_license.docx
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_train_rgb.zip.001
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_train_rgb.zip.002
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_train_rgb.zip.003
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_train_rgb.zip.004
cat dataset_train_rgb.zip* > dataset_train_rgb.zip
find . -type f -name 'dataset_train_rgb.zip.*' -delete
unzip dataset_train_rgb.zip
rm dataset_train_rgb.zip
rm non-commercial_license.docx
# download the pretrained model
#cd ~/models/model
echo -e "\n downloading pretrained models"
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
tar -xf ssd_mobilenet_v1_coco_11_06_2017.tar.gz 
rm ssd_mobilenet_v1_coco_11_06_2017.tar.gz 
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -xf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
rm faster_rcnn_resnet101_coco_11_06_2017.tar.gz
cd ..
echo -e "\n installing dependencies"
pip install tensorflow-gpu
pip install tqdm
echo -e "\n generating train tfrecords"
chmod +x "bosch-to-tfrecords.py"
./bosch-to-tfrecords.py --output_path=data/train.record
echo -e "\n generating test tfrecords"
chmod +x "bosch-to-tfrecords-test.py" 
./bosch-to-tfrecords-test.py --output_path=data/test.record

# set up protobuf
echo -e "\n set up protobuf"
# https://gist.github.com/sofyanhadia/37787e5ed098c97919b8c593f0ec44d8
curl -OL https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-linux-x86_64.zip
unzip protoc-3.4.0-linux-x86_64.zip
sudo mv bin/* /usr/local/bin/
rm -r bin
sudo mv include/* /usr/local/include/
rm -r include

# Protobuf compilation (object detection installation dependencies)
protoc object_detection/protos/*.proto --python_out=.
# add libraries to python path (object detection installation dependencies)
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
#Testing the Installation
python object_detection/builders/model_builder_test.py