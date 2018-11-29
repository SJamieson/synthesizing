# Set this to the path to Caffe installation on your system
caffe_root = "/path/to/your/caffe/python" 
gpu = True

cornet_s_weights = "nets/CORnet-S/Sequential.caffemodel"
cornet_s_definition = "nets/CORnet-S/Sequential.prototxt"
cornet_z_weights = "nets/CORnet-S/Sequential.caffemodel"
cornet_z_definition = "nets/CORnet-S/Sequential.prototxt"
caffenet_weights = "nets/caffenet/bvlc_reference_caffenet.caffemodel"
caffenet_definition = "nets/caffenet/caffenet.prototxt"

# -------------------------------------
# These settings should work by default
# DNN being visualized
# These two settings are default, and can be overriden in the act_max.py
net_weights = cornet_s_weights
net_definition = cornet_s_definition

# Generator DNN
generator_weights = "nets/upconv/fc6/generator.caffemodel"
generator_definition = "nets/upconv/fc6/generator.prototxt"

# Encoder DNN
encoder_weights = "nets/caffenet/bvlc_reference_caffenet.caffemodel"
encoder_definition = "nets/caffenet/caffenet.prototxt"
