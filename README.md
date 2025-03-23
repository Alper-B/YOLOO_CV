# YOLOO_CV
hackaton2025
- # Data preprocessing
- Only the images with masks made the final train dataset. However, we tried:
	- Apply many edge detection algorithm and adding the processed images as additional channels to the input images (r, g, b, canny)
	- We also tried to apply wavelet transform (space - frequency domain balance)
	- We tried incorporating the originals
		- We trained a model which takes an inpainted image and produces  the difference to the original one. Then we took the output of the model and used it as a separate input channel. This didn't improve performance either.
	- We also tried to saturate the images to hopefully emphasize artifacts.
	- So, at the end we used the original images. This performed the best. Our explanation is that this is due to the fact that the foundation models were trained on natural images.
	- Adding fourier transforms as separate features seemed to improve performance, however, we didn't have time to retrain our model
- # Model Design
- We used the DeepLab3+ architecture. It is an autoencoder architecture.
	- Our initial approach was a vanilla UNet. It is a standard architecture segmentation for segmentation. However, it was not powerful enough (our DCE validation never ).
	- The next idea was to swap the encoder for a foundation model. After searching around, we found DeepLab3+, which is a refinement of the above idea.
	- As backbone we used efficientnet-b7
- Training
	- We used a standard training loop - Adam optimizer with scheduler.
	- We employed early stop ,which saves the model that achieves the best validation score in the course of the training. The thing that performed best was to just let the model train for a very long time.
- Evaluation
	- We mainly compared average Dice scores of our different models to pick the best one.
 #Usage
	-main_submission.py is the main file that used to train the model.
	-eval.py is the python script that used to evaluate the models
