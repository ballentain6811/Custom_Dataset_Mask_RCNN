import sys, fileinput, os, subprocess
from Utils import modifyConfig, getTrainScript, getExportScript, resetEnv

# system path 추가
sys.path.append('D:\\TF_Object_Detection_API')
sys.path.append('D:\\TF_Object_Detection_API\\slim')

num_step = 1500
save_step = 500
current_step = save_step
train_dir = './retrained_model'
pipeline_config_path = './model_configs/mask_rcnn_resnet101_atrous_coco.config'

base_save_path = './exported_model/mask_rcnn_resnet101'
if not os.path.isdir(base_save_path):
    os.mkdir(base_save_path)

train_py_script = getTrainScript(train_dir, pipeline_config_path)

while current_step < num_step:
    print('[Step] : {}' .format(current_step))
    modifyConfig(pipeline_config_path, current_step)

    # retrain
    try:
        subprocess.run([exec(train_py_script)])
    except:
        print('에러를 완벽히 잡아내진 못했지만 실행에는 문제가 없음')
    resetEnv()

    # export
    export_py_script = getExportScript(pipeline_config_path, current_step, base_save_path)
    try:
        subprocess.run([exec(export_py_script)])
    except:
        print('에러를 완벽히 잡아내진 못했지만 실행에는 문제가 없음')
    resetEnv()
       
    current_step += save_step
    

print('[Step] : {}' .format(num_step))    
modifyConfig(pipeline_config_path, num_step)
try:
    subprocess.run([exec(train_py_script)])
except:
    print('에러를 완벽히 잡아내진 못했지만 실행에는 문제가 없음')
resetEnv()

export_py_script = getExportScript(pipeline_config_path, current_step, base_save_path)
try:
    subprocess.run([exec(export_py_script)])
except:
    print('에러를 완벽히 잡아내진 못했지만 실행에는 문제가 없음')
resetEnv()
