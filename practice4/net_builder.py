import caffe

def base_layer(layer_name, type_name, \
               bottom_names, top_names):
  layer = caffe.proto.caffe_pb2.LayerParameter()
  layer.name = layer_name
  layer.type = type_name
  layer.bottom.extend(bottom_names)
  layer.top.extend(top_names)
  return layer

def convolution_layer(bottom_name, top_name, \
      num_output, kernel_size, stride=1, pad=0, \
      group=1, bias_term=True):
  layer_name = top_name + '/conv'
  layer = base_layer(layer_name, 'Convolution', \
                    [bottom_name], [top_name])
  layer.convolution_param.num_output = num_output
  layer.convolution_param.kernel_size.append(kernel_size)
  layer.convolution_param.stride.append(stride)
  layer.convolution_param.pad.append(pad)
  layer.convolution_param.group = group
  layer.convolution_param.bias_term = bias_term
  layer.convolution_param.weight_filler.type = 'xavier'
  return layer

def data_layer(source, batch_size=25, \
               top_names=None, crop_size=227, \
               mean_values=None):
  if top_names is None:
    top_names = ['data', 'label']
  if mean_values is None:
    mean_values = [104, 117, 123]
  layer_name = top_names[0] + '/data'
  layer = base_layer(layer_name, 'Data', [], top_names)
  layer.transform_param.crop_size = crop_size
  layer.transform_param.mean_value.extend(mean_values)
  layer.data_param.source = source
  layer.data_param.batch_size = batch_size
  layer.data_param.backend = caffe.params.Data.LMDB
  return layer

def batch_norm_layer(bottom_name, top_name):
  layer_name = top_name + '/bn'
  layer = base_layer(layer_name, 'BatchNorm', [bottom_name], [top_name])
  return layer

def scale_layer(bottom_name, top_name, bias_term=True):
  layer_name = top_name + '/scale'
  layer = base_layer(layer_name, 'Scale', [bottom_name], [top_name])
  layer.scale_param.bias_term = bias_term
  return layer

def relu_layer(bottom_name, top_name):
  layer_name = top_name + '/relu'
  layer = base_layer(layer_name, 'ReLU', [bottom_name], [top_name])
  return layer

def softmax_loss_layer(bottom_names, top_name):
  layer_name = top_name + '/softmax_loss'
  layer = base_layer(layer_name, 'SoftmaxWithLoss', bottom_names, [top_name])
  return layer

def conv_module(bottom_name, top_name, \
                num_output, kernel_size, \
                stride=1, pad=0, group=1):
  module = caffe.proto.caffe_pb2.NetParameter()
  # Conv layer
  module.layer.extend( \
    [convolution_layer(bottom_name, top_name, \
          num_output, kernel_size, stride, pad, \
          group, bias_term=False)])
  # BatchNorm layer
  module.layer.extend( \
    [batch_norm_layer(top_name, top_name)])
  # Scale layer
  module.layer.extend( \
    [scale_layer(top_name, top_name, bias_term=True)])
  # ReLU layer
  module.layer.extend( \
    [relu_layer(top_name, top_name)])
  return module

def conv_net(names, channels, kernels, strides):
  net = caffe.proto.caffe_pb2.NetParameter()
  # Data layer
  net.layer.extend( \
    [data_layer('data/imagenet/train_lmdb', batch_size=32)])
  # Conv modules
  for i in range(len(channels)):
    pad = (kernels[i] - 1) / 2
    net.MergeFromString( \
      conv_module(names[i], names[i+1], \
               channels[i], kernels[i], strides[i], pad) \
      .SerializeToString())
  # Loss layer
  net.layer.extend( \
    [softmax_loss_layer(['out', 'label'], 'loss')])
  return net
