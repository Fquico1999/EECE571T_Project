# EECE571T_Project

## Artifact Segmentation

Currently, artifact segementation is implememted using a UNet Model that generates masks corresponding to artifact pixels.


To train a unet model run `python train_unet.py -i <path_to_images> -m <path_to_masks>`. This generates`unet_model` containing the saved model. By default, `batch=16` and `epochs=3`, which yielded about 97% accuracy on a small dataset.

To predict masks for new images, run `python predict_artifacts.py -i <path_to_images> -mp unet_model` which takes the trained unet model and runs the images through it. By default, masks are stored in `/predicted_masks`.

To remove artifacts from a set of images, run `python remove_artifacts.py -i <path_to_images> -mp unet_model`. Currently, the script will run the images through the model and obtain the artifact masks. Binary thresholding and connected component filtering are the current implementations to remove smaller scale artifacts that may actually be scored nuclei. By default, the processed images are stored in `/artifactless`

Settings regarding input image, model input, and patch dimensions are stored in `config.json`
