python train_cv_emp_time.py -net alexnet -epoch 5 -precision emp
python train_cv.py -net alexnet -epoch 5 -precision amp 
python train_cv.py -net alexnet -epoch 5 -precision fp32
python train_cv_emp_time.py -net vgg16 -epoch 5 -precision emp
python train_cv.py -net vgg16 -epoch 5 -precision amp
python train_cv.py -net vgg16 -epoch 5 -precision fp32
python train_cv_imagenet_emp_time.py -net alexnet -epoch 1 -precision emp
python train_cv_imagenet.py -net alexnet -epoch 1 -precision amp
python train_cv_imagenet.py -net alexnet -epoch 1 -precision fp32
python train_cv_imagenet_emp_time.py -net vgg16 -epoch 1 -precision emp -batch_size 128
python train_cv_imagenet.py -net vgg16 -epoch 1 -precision amp -batch_size 128
python train_cv_imagenet.py -net vgg16 -epoch 1 -precision fp32 -batch_size 128