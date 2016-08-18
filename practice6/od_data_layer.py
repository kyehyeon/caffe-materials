from caffe import Layer
import numpy as np
import cv2
import xml.etree.ElementTree as et
import json
import os

###################################
# XML Parsing
###################################

def parse_value(node, key):
  value = node.find(key).text
  return int(round(float(value)))

def parse_object(node):
  label = ''
  box = []
  for child in node:
    if child.tag == 'name':
      label = child.text
    elif child.tag == 'bndbox':
      box = [parse_value(child, 'xmin'), \
             parse_value(child, 'ymin'), \
             parse_value(child, 'xmax'), \
             parse_value(child, 'ymax')]
  return (label, box)

def parse_root(root):
  folder = ''
  filename = ''
  objs = []
  for child in root:
    if child.tag == 'folder':
      folder = child.text
    elif child.tag == 'filename':
      filename = child.text
    elif child.tag == 'object':
      objs.append(parse_object(child))
  return (folder, filename, objs)

def parse(filename):
  tree = et.parse(filename)
  root = tree.getroot()
  return parse_root(root)


###################################
# Object classes
###################################

labels = ('__background__', \
  'bicycle', 'bird', 'bus', 'car', 'cat', \
  'dog', 'horse', 'motorbike', 'person', 'train', \
  'aeroplane', 'boat', 'bottle', 'chair', 'cow', \
  'diningtable', 'pottedplant', 'sheep', 'sofa', \
  'tvmonitor')
label_dict = { label: i \
               for i, label in enumerate(labels) }


###################################
# Data layer
###################################

class ODDataLayer(Layer):
  def setup(self, bottom, top):
    params = json.loads(self.param_str)
    self.source_file = params['source']
    self.source = open(self.source_file, 'r')
    self.img_dir = params['img_dir']
    self.mean = params['mean']
    self.base_size = params['base_size']
    max_size = max(self.base_size) / 32 * 32
    top[0].reshape(1, 3, max_size, max_size)
    top[1].reshape(1, 6)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    # Read 1 .xml filename from the source list file
    line = self.source.readline().strip()

    # After reading the last .xml, go back to the first .xml
    if line is None or len(line) == 0:
      self.source.close()
      self.source = open(self.source_file, 'r')
      line = self.source.readline().strip()

    # Parse it
    fdir, fname, objs = parse(line)

    path = os.path.join(self.img_dir, fname)
    img = cv2.imread(path)

    # Choose base size randomly
    random_idx = np.random.randint(len(self.base_size))
    base_size = float(self.base_size[random_idx])

    # Compute scaleed image size
    short_side = min(img.shape[0], img.shape[1])
    scale = base_size / short_side
    scale_h = int(img.shape[0] * scale / 32) * 32.0 / img.shape[0]
    scale_w = int(img.shape[1] * scale / 32) * 32.0 / img.shape[1]
    scale_xyxy = np.array([scale_w, scale_h, scale_w, scale_h])
    h = int(round(img.shape[0] * scale_h))
    w = int(round(img.shape[1] * scale_w))

    # Rescale image
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # height x width x 3  -->  3 x height x width
    img = np.rollaxis(img, 2, 0)

    # Subtract mean
    img = np.array(img, dtype=np.float32)
    img[0,:,:] -= self.mean[0]
    img[1,:,:] -= self.mean[1]
    img[2,:,:] -= self.mean[2]

    # Assign the pre-processed image to network input
    top[0].reshape(1, 3, img.shape[1], img.shape[2])
    top[0].data[...] = img

    # Assign label information
    top[1].reshape(len(objs), 6)
    for n, (label, box) in enumerate(objs):
      # Since batch size = 1, batch item index is always 0
      top[1].data[n, 0] = 0
      # Rescale true box
      box = np.array(box, dtype=np.float32) * scale_xyxy
      top[1].data[n, 1:5] = box
      # True class (not used in this lecture)
      top[1].data[n, 5] = label_dict[label]

  def backward(self, top, propagate_down, bottom):
    pass
