import cv2, os, io
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

def image_to_tf_data(img_path, mask_path, class_name):
    filename = img_path.split('/')[-1]

    # image
    with tf.gfile.GFile(img_path, 'rb') as fid: # 이미지를 binary mode로 읽음
        encoded_img = fid.read()
    encoded_img_np = np.fromstring(encoded_img, dtype = np.uint8)
    image = cv2.imdecode(encoded_img_np, cv2.IMREAD_COLOR)
    height, width, _ = image.shape

    # mask
    with tf.gfile.GFile(mask_path, 'rb') as fid:
        encoded_mask = fid.read()
    encoded_mask_np = np.fromstring(encoded_mask, dtype = np.uint8)
    mask = cv2.imdecode(encoded_mask_np, cv2.IMREAD_GRAYSCALE) # mask는 1차원으로 사용함
    mask_pixel_vals = []
    for val in range(1, 256):
        if val in mask:
            mask_pixel_vals.append(val)

    # 필요 데이터들 추출 뒤 저장할 변수들
    classes, classes_text = [], []
    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    encoded_mask_png_list = []
    for pixel_val in mask_pixel_vals:
        class_idx = 1

        # xmin, ymin, xmax, ymax 찾는 과정
        check_x_coordi = np.any(mask == pixel_val, axis = 0)
        check_y_coordi = np.any(mask == pixel_val, axis = 1)
        object_x_coordi = np.where(check_x_coordi)[0]
        object_y_coordi = np.where(check_y_coordi)[0]

        xmin = min(object_x_coordi)
        xmax = max(object_x_coordi)
        ymin = min(object_y_coordi)
        ymax = max(object_y_coordi)

        object_mask = np.uint8(mask == pixel_val) # mask의 최대값은 1 이다
        encoded_mask_png = cv2.imencode('.PNG', object_mask)[1].tobytes()  # mask는 PNG 확장자로 저장해줘야하는 듯

        classes.append(class_idx)
        classes_text.append(class_name.encode('utf8'))
        xmins.append(xmin / width)
        ymins.append(ymin / height)
        xmaxs.append(xmax / width)
        ymaxs.append(ymax / height)
        encoded_mask_png_list.append(encoded_mask_png)

    # TFRecord protocol 같은 부분임
    # 여기에 맞춰서 데이터 만들어서 넣어주면 됨
    feature_dict = {    
        'image/height':             dataset_util.int64_feature(height),
        'image/width':              dataset_util.int64_feature(width),
        'image/filename':           dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':          dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256':         dataset_util.bytes_feature(class_name.encode('utf8')),
        'image/encoded':            dataset_util.bytes_feature(encoded_img),
        'image/format':             dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin':   dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax':   dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin':   dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax':   dataset_util.float_list_feature(ymaxs),
        'image/object/class/text':  dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/mask':        dataset_util.bytes_list_feature(encoded_mask_png_list)}
    tf_data = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return tf_data

base_dir = './Dataset/valid/images'
img_names = os.listdir(base_dir)
img_paths = ['{}/{}' .format(base_dir, name) for name in img_names]

tf_writer = tf.python_io.TFRecordWriter('./Dataset/valid.record')
for img_path in img_paths:
    print(img_path)
    mask_path = img_path.replace('/images', '/masks')
    mask_path = mask_path.replace('_image', '_mask')

    tf_example = image_to_tf_data(img_path, mask_path, 'instrument')
    
    tf_writer.write(tf_example.SerializeToString())
tf_writer.close()



