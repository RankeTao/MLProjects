import os 
import cv2
import json
import numpy as np
import tensorflow as tf


def class_map(json_file, mode='label2name'):
    '''
        json_file: \n
        mode(str): {'label2name', 'name2label'} \n
        return a dict mapping.
    '''
    with open(jsn_file_path, 'r', encoding='utf-8') as meta_file:
        meta_data = json.load(meta_file)
    print(meta_data['class_detail'])
    class_map = {}
    for i in range(len(meta_data['class_detail'])):
        class_map[meta_data['class_detail'][i]["class_label"]] = \
            meta_data['class_detail'][i]["class_name"]
    if mode == 'label2name':
        pass
    else:
        class_map = {v : k for k, v in class_map.items()}
    print(class_map)
    return class_map


def img_preprocess(file, shape=None, color='RGB'):
    '''
    file(str): an image file.\n
    shape(int): resize current image size(height, width) to (shape, shape).\n
    color(str): default color is 'RGB'.
    '''
    img= cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if color == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if shape != None:
        assert isinstance(shape, int)
        img = cv2.resize(img, (shape, shape))
        img = img.astype(np.float32)/255.0  #图像数据一定要归一化处理，不然预测很不准确
    return img


def images2pred(dir, shape=32):
    assert os.path.isdir(dir)
    img_name_lst = [item for item in os.listdir(dir) 
                            if os.path.isfile(os.path.join(dir, item))]
    image_array = np.zeros((len(img_name_lst), shape, shape, 3))
    for i in range(len(img_name_lst)):
        image_file = os.path.join(dir, img_name_lst[i])
        image = img_preprocess(image_file, shape=shape)
        image_array[i] = image
    return image_array, img_name_lst


if __name__ == '__main__':
    # create class map according to the json file
    jsn_file_path = "face/readme.json"
    class_map = class_map(jsn_file_path, mode='label2name')
    
    image_dir = r"C:\Users\iweut\PycharmProjects\face-recognition-baidu\images\predict"
    image_array, img_name_lst = images2pred(image_dir, shape=32)
    print(image_array.shape)
    
    # load the model
    model = tf.keras.models.load_model("lenet.h5")
    prediction = model.predict(image_array) # prediction returns an array of (number_of_images, len(class))

    print(prediction)  # The first image results
    print(np.argmax(prediction[0]))
    for i in range(len(img_name_lst)):
        if np.any(prediction[i] > 0.80):
            pred_name = class_map[np.argmax(prediction[i])]
            print(f"The prediction of image '{img_name_lst[i]}' is '{pred_name}'")
        else:
            print("The image '{}' is not one of Catergaries!".format(img_name_lst[i]))