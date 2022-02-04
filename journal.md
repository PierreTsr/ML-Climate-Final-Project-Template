# Project Journal

## 01/22/2022

- First look at the code base from DataClinic (https://github.com/tsdataclinic/PMLDataPipeline).
- Failed to run their code due to version compatibility issues. 
- Search for fixes + Issue submission on their repo.
- Seems like the training data is not easily accessible from their repo, they only provide trained models.

## 01/27/2022

- Read about the implementation of their code (https://www.youtube.com/watch?v=ylT4J6wCctQ).
- They have quite minimal data (~ 1e2 pixels with identified platic debris, in a few specific regions), their goal was to use the trained model to enlarge that dataset, but the project seems to be on halt since 07/2020.
- I parsed the internet for plastic debris datasets. None is using Sentinel-2 imagery, but I have identified a few promising leads:
  - Use labeled imagery from another satellite and cross-reference the time and coordinates to find the corresponding Sentinel-2 images (https://mlhub.earth/data/nasa_marine_debris);
  - Use GPS + timed data from beach clean-up operations to identify a set of images with suspected plastic debris (https://www.kaggle.com/maartenvandevelde/marine-litter-watch-19502021, https://www.cefas.co.uk/data-and-publications/dois/clip-belize-marine-litter-abundance-and-composition-2019/);
- Rewrite the abstract with new set of goals/milestones.
- Read EO-Learn documentation to prepare for a large update of their code.

## 02/03/2022

- After discussing with Pr. Kucukelbir, realized that Sentinel-Hub subscription could be avoided.
- Changed the code to use manually downloaded tiles. It requires quite a bit work for now:
  1. Downloading tiles in `.SAFE` format from https://scihub.copernicus.eu/dhus/#/home
  2. Using ESA's SNAP tool to resample the tiles to the desired resolution, and exporting them to `GeoTIFF` format (approx. 1.5Gb/tile)
  3. Modifying the `.json` description file to use the new coordinates
- It should be possible to use ESA's API and a bash script to streamline the first two steps, but I'll see that later. Now the code is fully operational (they were a lot of outdated/errored parts) and able to process efficiently a tile to compute our input features. I then need to have look at the inference part. I still haven't looked for the labeled data.
