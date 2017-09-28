# usage python bosch-to-tfrecords.py --output_path=train.record
import tensorflow as tf
from models.research.object_detection.utils import dataset_util
from tqdm import tqdm
import yaml
import os


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

label_map={
  'Yellow' : 1,
  'RedLeft' : 2,
  'Red' : 3,
  'GreenLeft' : 4,
  'Green': 5,
  'off' : 6,
  'GreenRight' : 7,
  'GreenStraight' : 8,
  'GreenStraightRight' : 9,
  'RedRight' : 10,
  'RedStraight' : 11,
  'RedStraightLeft' : 12,
  'GreenStraightLeft' : 13
}

def create_tf_example(example):
  DIR = 'data/rgb/test' #sim_training_data_large/'
  height = 720 # Image height
  width = 1280 # Image width
  filename = str.encode(os.path.join(os.path.abspath(''), DIR, os.path.basename(example['path'])))    
  
  with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image_data = fid.read() # Encoded image bytes
  image_format = 'png'.encode()
  xmins, xmaxs, ymins, ymaxs, classes_text, classes = [],[],[],[],[],[]
  for box in example['boxes']:

    xmins.append(box['x_min']*1./width) # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs.append(box['x_max']*1./width) # List of normalized right x coordinates in bounding box
               # (1 per box)
    ymins.append(box['y_min']*1./height) # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs.append(box['y_max']*1./height) # List of normalized bottom y coordinates in bounding box
               # (1 per box)
    label = str.encode(box['label'])
    
    classes_text.append(str.encode(box['label'])) # List of string class name of bounding box (1 per box)
    classes.append(label_map[box['label']]) # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  with open('data/test.yaml', 'r') as stream:
    try:
      print ("Generating TFRecords")
      examples=yaml.load(stream)
      for elem in tqdm(examples):  
        tf_example = create_tf_example(elem)
        writer.write(tf_example.SerializeToString())
    except yaml.YAMLError as exc:
      print(exc)

  print ("TFRecords has been generated successfully")
  writer.close()

if __name__ == '__main__':
  tf.app.run()