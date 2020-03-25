import sys, fileinput, os
import tensorflow as tf

def modifyConfig(pipeline_config_path, step):
    for line in fileinput.input(pipeline_config_path, inplace = True):
        if 'num_steps: ' in line:
            line = line.replace(line, '  num_steps: {}\n' .format(str(step)))
        sys.stdout.write(line)

def getTrainScript(train_dir, pipeline_config_path):
    script = open('./object_detection/legacy/train.py').read()
    script = script.replace("flags.DEFINE_string('train_dir', ''",
                            "flags.DEFINE_string('train_dir', '{}'" .format(train_dir))
    script = script.replace("flags.DEFINE_string('pipeline_config_path', ''",
                            "flags.DEFINE_string('pipeline_config_path', '{}'" .format(pipeline_config_path))

    return script

def getExportScript(pipeline_config_path, step, base_save_path):
    trained_checkpoint_prefix = './retrained_model/model.ckpt-' + str(step)
    output_directory = '{}/Step_{}' .format(base_save_path, str(step))
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    
    script = open('./object_detection/export_inference_graph.py').read()
    
    script = script.replace("flags.DEFINE_string('pipeline_config_path', None",
                            "flags.DEFINE_string('pipeline_config_path', '{}'" .format(pipeline_config_path))
    script = script.replace("flags.DEFINE_string('trained_checkpoint_prefix', None",
                            "flags.DEFINE_string('trained_checkpoint_prefix', '{}'" .format(trained_checkpoint_prefix))
    script = script.replace("flags.DEFINE_string('output_directory', None",
                            "flags.DEFINE_string('output_directory', '{}'" .format(output_directory))
    
    return script

def resetEnv():
    '''
     한 개의 프로세스(?)가 계속 진행되는 거라

     같은 변수 또는 오퍼레이션 등을 반복해서 선언하면 중복돼서 에러 발생함.

     이를 잡아주기 위해 초기화 시킴.
     
    '''
    
    # retrain 과정에서 초기화 해줘야됨
    keys = list(tf.app.flags.FLAGS._flags().keys())
    for key in keys:
        tf.app.flags.FLAGS.__delattr__(key)

    # export 과정에서 초기화 해줘야됨
    tf.reset_default_graph()

