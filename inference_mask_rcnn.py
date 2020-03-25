import numpy as np
import sys, os, cv2, time


# tensorflow version 확인하고 맞지 않으면 예외발생
import tensorflow as tf
from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('ver1.9.* 보다 높은 version의 tensorflow를 사용하세요.')

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()

            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']: 
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)         

            # 추가
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(tensor_dict, feed_dict = {image_tensor : [image]})
            
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

            return output_dict

def detection_result_only_one(img, output_dict, image_save_path, save = False):
    mask = output_dict['detection_masks'][0]
    mask = np.uint8(mask * 80)
    mask = cv2.merge((np.zeros_like(mask), mask, np.zeros_like(mask)))
    img = cv2.add(img, mask)
    
    if save:
        cv2.imwrite(image_save_path, img)
    else:
        cv2.imshow('Result', img)
        cv2.waitKey(1)

    
###### hyper-parameter 설정 ############################################################


model_path = './exported_model/mask_rcnn_resnet101/Step_500/frozen_inference_graph.pb'
base_save_path = './Results/mask_rcnn_resnet101/Step_500/'


#########################################################################################

# model load
detection_graph = tf.Graph()
with detection_graph.as_default():                         
    od_graph_def = tf.GraphDef()                                                                            
    with tf.gfile.GFile(model_path, 'rb') as fid:                                                            
        serialized_graph = fid.read()                              
        od_graph_def.ParseFromString(serialized_graph)                                                                 
        tf.import_graph_def(od_graph_def, name = "")    
print('계산 그래프 설정 완료...')

# inference and save
base_dir = './Dataset/valid/images'
image_names = os.listdir(base_dir)
image_paths = ['{}/{}' .format(base_dir, name) for name in image_names]
for image_path in image_paths:
    save_name = image_path.split('/')[-1]
    image = cv2.imread(image_path)

    time1 = time.time()
    output_dict = run_inference_for_single_image(image, detection_graph)
    print('[ {name} ] Sec : {sec:.3f}' .format(name = save_name, sec = time.time() - time1))
    
    save_path = base_save_path + save_name
    detection_result_only_one(image, output_dict, save_path, save = True)
    
        
print('저장 완료..')
        
    

    
    



















                                            
                                            
