import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

# [B, H, W, C]
def VGG(feature, im_height, im_width, num_classes):
    input_image = layers.Input(shape = (im_height, im_width, 3), dtype = "float32")
    x = feature(input_image) # 特征提取
    x = layers.Flatten()(x)
    x = layers.Dropout(rate = 0.5)(x)
    x = layers.Dense(4096, activation = 'relu')(x)
    x = layers.Dropout(rate = 0.5)(x)
    x = layers.Dense(4096, activation = 'relu')(x)
    x = layers.Dense(num_classes)(x)
    output = layers.Softmax()(x)
    model = Model(inputs = input_image, outputs = output)
    return model

def features(cfg):
    feature_layers = [] # 存储对应网络的网络层
    for v in cfg:
        if v == "M":
            feature_layers.append(layers.MaxPool2D(pool_size = 2, strides = 2)) # VGG网络中每一层池化层的池化核大小和步距都是一样的
        else:
            feature_layers.append(layers.Conv2D(v, kernel_size = 3, padding = "SAME", activation = "relu")) # VGG网络中每一层卷积层的卷积核大小和步距都是一样的
    return Sequential(feature_layers, name = "feature")

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # 数字表示卷积核个数， M表示最大池化下采样
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def VggNet(model_name = "vgg16", im_height = 224, im_width = 224, num_classes = 5):
    assert model_name in cfgs.keys(), "not support model {}".format(model_name)
    cfg = cfgs[model_name]
    model = VGG(features(cfg), im_height, im_width, num_classes)
    return model

if __name__ == "__main__":
    input1 = tf.random.normal([32, 224, 224, 3])
    model = VggNet(model_name = "vgg16")
    output = model(input1)
    print(output)