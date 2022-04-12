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

## 02/10/2022

- Reached out to Dr. Laura Biermann to request access to labeled data (no answer so far) -> In the mean time focussing on unsupervised approaches.
- Spotted a few insidious errors in the code base (physical parameters were hard-coded with incorrect values).
- Tried a few unsupervised approaches on small patches, but faced some problem specific issues (cf. https://edstem.org/us/courses/17799/discussion/1150842)
- Exploratory analysis with enhanced visuals show that it should be possible to isolate only the debris. This should provide a reduces and more balanced dateset to use unsupervised learning on.

## 2022-03-07 check in: alp

Looking good. Would encourage trying to get some initial results soon. Also please try to update this weekly.

## 23/03/2022
I forgot to update the journal in a while due to the midterms and the break, but here is what happened since last time:

- I found a solution to the issue I posted on EdStem: I am using robust covariance estimates to isolate the outliers (i.e. the debris) in the tiles. I have experienced with a few variations around this method and I am now able to isolate all the debris fairly reliably. I will use the labeled data I just recieved to provide some quantified results (my goal will be to maximize recall, as the aim of this part is to reduced the quantity of water pixels);
- I also created an entirely new visualization workflow to help me in the exploratory analysis;
- I tried a few out-of-the-box unsupervised clustering algorithm from `sklearn`. But, I wasn't able to evaluate them without the labeled data;
- I just recieved some labeled data from Dr. Biermann, I am still querying the related tiles from the ESA portal and pre-processing them;
- In the coming weeks I will have more time to put in the project, and my goal will be to create a visualization tool to explore the unsupervised classification results, and apply it to the labeled data to get some quantified results.

## 02/04/2022

- I had to reshape some part of my pipeline to make it usable with my limited ressources (still the same issue: a tile has too many datapoints, and I need an efficient way to filter them);
- Now, given a set of observations, once the tile is downloaded, I can produce a set of unsupervised identified debris in the same location in a few minutes. I still need to make some progess on the classification part, but it should make it much easier to compare my results with labeled data;
- I'm currently working on a notebook with an interactive display of those results;
- The next steps are: 
  - using this pipeline on more observations to have quatified estimates of its recall;
  - trying with different clustering/features combination to find the most accurate one;
  - quantifying these results;

## 08/04/2022

- I started using the MARIDA dataset, which is immensly useuful. I am now able to process any tile in the dataset (after querying them on the ESA's platform), find all the outliers, and merge in the annotations from the labeled scene in MARIDA. All of that in a few minutes;
- I am currently working on a subset of 9 out of 63 tiles from the dataset:
  - I have a recall of 1.0 on all the interesting categories with my outlier identification (no-classification at this stage), while exclusing between 99% and 99.9% of the pixels, depending on the method and tile;
  - Unsupervised classification will likely be very hard, as the distribution is highly continuous;
  - On the bright side, MARIDA provides enough data so that unsupervised learning might not be the best suited approach anymore;
  - I trained a few very simple supervised models, and it seems promising: around 80% F1 score so far. This part needs to be pushed further;
- From there, the next steps are:
  - Collect the entire dataset;
  - Provide an evaluation of a Gaussian Naive Bayes model (as Dr. Biermann's paper);
  - Provide an evaluation of a few Random Forest models (as MARIDA's paper);
  - Compare the results with and without the gaussian normalization of pixel values and indices;

