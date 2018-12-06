import numpy as np
import caffe

ilsvrc2012_mean = np.array((104.0, 117.0, 123.0))  # ImageNet Mean in BGR order

net_io_layers = {'deepsim-norm1': {'input_layer_shape':  (96, 27, 27,),
                                   'output_layer_shape': (3, 240, 240,)},
                 'deepsim-norm2': {'input_layer_shape':  (256, 13, 13,),
                                   'output_layer_shape': (3, 240, 240,)},
                 'deepsim-conv3': {'input_layer_shape':  (384, 13, 13,),
                                   'output_layer_shape': (3, 256, 256,)},
                 'deepsim-conv4': {'input_layer_shape':  (384, 13, 13,),
                                   'output_layer_shape': (3, 256, 256,)},
                 'deepsim-pool5': {'input_layer_shape':  (256, 6, 6,),
                                   'output_layer_shape': (3, 256, 256,)},
                 'deepsim-fc6':   {'input_layer_shape':  (4096,),
                                   'output_layer_shape': (3, 256, 256,)},
                 'deepsim-fc7':   {'input_layer_shape':  (4096,),
                                   'output_layer_shape': (3, 256, 256,)},
                 'generator':     {'input_layer_shape':  (4096,),
                                   'output_layer_shape': (3, 224, 224,)},
                 'deepsim-fc8':   {'input_layer_shape':  (1000,),
                                   'output_layer_shape': (3, 256, 256,)}}


def get_transformer(net_name='generator', scale=255):
    transformer = caffe.io.Transformer({'data': (1, 3, 224, 224)})
    transformer.set_transpose('data', (2, 0, 1))  # move color channels to outermost dimension
    transformer.set_raw_scale('data', scale)  # should be set if pixel values are in [0, scale], not [0, 1]
    transformer.set_mean('data', ilsvrc2012_mean / (255 / scale))  # subtract the dataset-mean value in each channel
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    return transformer

def visualize(generator, code, transformer, digitize=True):
    output_size = (256, 256)
    image_size = (224, 224)
    topleft = ((output_size[0] - image_size[0]) / 2, (output_size[1] - image_size[1]) / 2)
    x = generator.forward(feat=code)['deconv0']   # names for io layers may need to be changed
    x = x[:, :, topleft[0]:topleft[0] + image_size[0], topleft[1]:topleft[1] + image_size[1]]
    x = x[0]
    print(x)
    x = x - np.mean(x, axis=(1,2)).reshape((3,1,1)) + ilsvrc2012_mean.reshape((3,1,1))
    x = x / 255
    x = x[::-1, :, :]
    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0, 1) * 255
    print(x)
    # x = x[14:241, 14:241, :]    # for clipping margins, if you want
    if digitize:
        return x.astype('uint8')
    else:
        return x