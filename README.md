# Monocular Depth Estimation to Predict Topology of the Martian Surface

![alt text](https://www.ox.ac.uk/sites/files/oxford/styles/ow_medium_feature/s3/field/field_image_main/Banner%20image%20resized.jpg?itok=Zp2eLA3N)

Part of the paper - [An Adversarial Generative Network Designed for High-Resolution Monocular Depth Estimation from 2D HiRISE Images of Mars](https://www.mdpi.com/2072-4292/14/18/4619)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mattiagatti/mars_dtm_estimation)

Monocular depth estimation is the task of estimating the depth value (distance relative to the camera) of each pixel, given a single (monocular) image. Typically, the depth values of an image are obtained using stereoscopic algorithms, which require the use of a stereoscopic camera (a camera with two or more lenses). Digital Terrain Models (DTMs) are essential for planetary exploration and scientific research. They are digital representations of the elevations of a surface, from which it is possible to reconstruct a three-dimensional model of the area. The High Resolution Imaging Science Experiment (HiRISE) is a camera on board the Mars Reconnaissance Orbiter, which has been orbiting and studying Mars since 2006. By using its images of the same area on the ground, taken from different angles, many DTMs of some parts of the surface have been created. However, creating a DTM is complicated and requires sophisticated software and a lot of time, both computational and human. This project will investigate the use of monocular depth estimation models to predict the topology of Mars in a faster and cheaper way.

## Disclaimer
The model performs monocular depth estimation and gives predictions in meters. In addition, the model was modified after the paper was published and the results are now better. In the paper another model was discussed, which increases input resolution by 4x and, at the same time, estimate a DTM with predictions in the range [0, 1]. You can find the code for that model [here](https://gitlab.com/riccardo2468/srdinet).

## Dataset
The dataset was scraped from the [UAHiRISE](https://https://www.uahirise.org/) website. The associated 1m/px 8-bit greyscale raster image (left or right image of the stereo pair) and Float32 raster DTM with the same spatial resolution were downloaded for each observation. The complexity of deep learning models is proportional to input size, and these rasters could reach very large shapes. Therefore, the image was split into smaller tiles, which will be merged after processing. In total, 160k patches were generated. Each patch is a 2-channel 512 x 512 raster, where the first channel is the image and the second channel is a DTM. The DTMs are stored as absolute values. However, the absolute heights must be converted to relative heights for use with a monocular depth estimation model. The dataset size is 500 GB.

## Commands

Initialize the environment:
``` console
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.11 python3.11-venv
cd hirise-monocular-depth-estimation
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Scrap mars data from [UAHiRISE](https://https://www.uahirise.org/):
``` console
nohup python scrape.py > download.log 2>&1 &
```

Divide generated tiles in train, validation and test splits:
``` console
python splits/generate_splits.py
```

Once you have downloaded the backbone weights from [here](https://deci-pretrained-models.s3.amazonaws.com/mit_backbones/mit_b4.pth), place the file in the glpdepth/weights folder and then train the model:
``` console
nohup python train.py --batch_size 4 > train.log 2>&1 &
```

Try the demo:
``` console
python app.py
```

Build a docker image
``` console
docker build -f ./API_App/Dockerfile --tag marsdem .
```
