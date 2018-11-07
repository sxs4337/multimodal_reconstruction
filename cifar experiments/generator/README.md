# tf-DeepSim
Tensorflow Implementation of DeepSim

## To start training
python main.py --dataRoot /home/as6520/dummy-net --encoderRoot /home/as6520/tf-DeepSim/model_weights/ --expDir None --numIters 1 --display 50 --adversarialWeight 1.0 --reconstructionWeight 1.0 --pixelwiseWeight 1.0 --enLr 0.1 --genLr 0.1 --disLr 0.1 --gpu_id 0
