1. run "pip install -r requirements.txt" to install all the requirements
2. git clone https://github.com/facebookresearch/dinov2.git 
3. wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth -P /content/pretrain
4. run "python inference.py --image_path[image_path]" to get inference result from single image
5. run "python class_accuracy.py" to get accuracy for every class
6. run "python random_class_inference.py" to get 4 random example from each class
