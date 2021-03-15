import tensorflow as tf
import numpy as np
from tqdm import tqdm

def load_inception_v3():
    #include_top = whether to include the fully-connected layer at the top, as the last layer of the network. Default to True.
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    # After the last conv. layer in the V3 model the shape of the output is 8x8x2048. 
    # Rest has not been included
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    return image_features_extract_model

def get_image_dataset(image_names):
    encode_train = sorted(set(image_names))
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    #image_dataset is an iterable of images, reads images here

    image_dataset = image_dataset.map(load_image, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16) #batch(multi-element) image_dataset transformations
    print("Shape of input images: ", image_dataset)

    return image_dataset

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def get_image_features(image_dataset, image_features_extract_model):
    for img, path in tqdm(image_dataset):
        #img shape: (16, 299, 299, 3) which the V3 model expects. Format: (batch, img shape)
        batch_features = image_features_extract_model(img) #shape: (16, 8, 8, 2048)
        batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, batch_features.shape[3])) #shape: (16, 64, 2048)

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

def extract_image_features(image_path, all_image_names):
    image_features_extract_model = load_inception_v3()
    image_dataset = get_image_dataset(all_image_names)
    get_image_features(image_dataset, image_features_extract_model)
