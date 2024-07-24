# Identifying Atmospheric Rivers on the West Coast of the United States with Geostationary Operational Environmental Satellite (GOES) Imagery

# Summary
This repository contains all necessary information (rationale, methodology, data, codes and results) used in identifying atmospheric rivers (ARs) in the Western United States.
The primary objective of this project is to explore the potential of GOES satellite data in accurately identifying ARs along the Pacific coast of North America. We developed and tested two machine learning models—a U-Net model and a random forest model—to identify ARs from GOES satellite imagery spanning the period from 2018 to 2020. 
The research questions guiding this investigation include: 
- Can optical satellite data from GOES, with its unique spatio-temporal resolution, be effectively utilized for the detection of ARs? 
- How do the performance of a high-fidelity U-Net model compare to those of a lower-fidelity Random Forest model in identifying ARs from GOES imagery?

# Methodology
# Study Area
Our research focuses on the Pacific West Coast of North America, as illustrated in Figures 1 (a) and 1 (b). The study area encompasses a broad region from the state of Washington in the north to California in the south, extending westward over the Pacific Ocean. 

![image](https://github.com/user-attachments/assets/d2015e82-fa22-4d5d-89f4-60e9e6776329)

Figure 1: Study Area and GOES Satellite Imagery. (a) Pacific West Coast region including the US, Canada, and Mexico. (b) GOES satellite image capturing atmospheric conditions over the study area (false color composite using bands 2,3,1).

# Dataset
# GOES Imagery
- This study utilizes data from the GOES-R series, specifically employing the Advanced Baseline Imager (ABI) with 16 channels, offering a spatial resolution of 2 km for most channels and a temporal resolution of 15 minutes for full disk scans. Data retrieval for the GOES-17 and GOES-18 satellites from 2016 to present is facilitated by the GOES-2-go Python package (Version 2022.07.15), which accesses data through Amazon Web Services (AWS) as part of NOAA's Open Data Dissemination Program [GOES2Go](https://github.com/blaylockbk/goes2go). package in github.
- The raw data from the GOES satellites required conversion into a standardized projection format. This step involved transforming the native GOES projection into a rectilinear latitude-longitude projection to ensure consistency with the analysis requirements and facilitate accurate spatial referencing. Additionally, the GOES images were resized from their original resolution of 4500x4996 pixels to 512x512 pixels to facilitate machine learning training and improve computational efficiency.

# Labeled data
- The dataset consists of continuous labeled data from 1979 to 2022, recorded at 6-hour intervals. The data was sourced from [Rhodes](https://portal.nersc.gov/archive/home/a/arhoades/Shared/www/TE_ERA5_ARs). which utilized the advanced TempestExtremes v2.1 algorithm to detect these features. The paper associated can be found [here](https://gmd.copernicus.org/articles/14/5023/2021/).

# ERA5 climate reanalysis product
- The retrieved data from ERA5 include wind components (u, v) and specific humidity (q) at multiple pressure levels (300 hPa to 1000 hPa). These data were used to calculate Integrated Water Vapor (IWV) and Integrated Vapor Transport (IVT). We utilized ERA-5 reanalysis data to calculate IVT and IWV and developed a thresholding algorithm to identify AR objects. This algorithm applies specific thresholds for IVT and IWV and evaluates the minimum length and width of potential AR regions to meet typical AR criteria. The algorithm starts by applying threshold values for IVT and IWV to create an initial mask of potential AR regions. Regions meeting these thresholds are marked. Minimum length and width requirements for ARs are converted from kilometers to grid points, ensuring precise identification of contiguous AR regions. Each connected region in the initial mask is labeled and evaluated against these criteria, resulting in a robust final binary mask representing AR regions. This mask is used for further analysis and benchmarking, providing a reliable method for identifying ARs in the ERA-5 reanalysis dataset.

# Models
# U-Net CNN architecture
- Learning rate: initialized to 0.001
- Batch size: 16
- Optimizer: Adam optimizer
- Loss function: Binary Cross-Entropy  and Focal Cross-Entropy (combined)
- Training: over 100 epochs
- Trained for 33 hours

![image](https://github.com/user-attachments/assets/4b5bb6f2-1aae-4d9a-acc1-dcbc4c50d14d)

Figure 2: Overall Methodological Framework

# RF model
- Decision trees: 100
- Maximum depth: none
- Max_features: sqrt
- Trained for 7 hours

# Results

# ERA-5 Benchmark Dataset

![image](https://github.com/user-attachments/assets/923efc07-4c7c-449c-8ebd-a62b4602fc71)

Figure 3: Atmospheric River Identification for December 13, 1997, at 12:00 UTC. (a) Integrated Vapor Transport (IVT), (b) Integrated Water Vapor (IWV), and (c) AR mask indicating identified atmospheric rivers.

# Image Segmentation using RF

![image](https://github.com/user-attachments/assets/170a86c8-f0c3-42e8-8a0c-e5b01028e39f)

Figure 4: Random Forest (RF) model predictions compared with labeled data and evaluation results. The top row shows (a) RF model prediction, (b) labeled data, and (c) evaluation of the model which has an IoU of 0.34 and an accuracy of 0.94. 

# Image Segmentation using CNN

![image](https://github.com/user-attachments/assets/5fd2b570-de8a-4ea4-afab-7216d8a11555)

Figure 5: AR identification using our U-Net model trained on GOES imagery, showing (a) brightness temperature from the water vapor bands (Bands 8-10) as a False Color Composite showing complex cloud patterns and atmospheric features, (b) The labeled dataset provided for training, highlighting the atmospheric river region in white, and (c) the prediction made by the U-Net model, indicating the probability of an atmospheric river (AR), where lighter shades represent higher probabilities.

# Codes
All codes that we used in the project is inside the folder "jupyter_notebooks"


# Report
- Report can be found here. (To be included)
- Supplemantary material can be found here. (To be included)
  
# References
Reviewed literature is tracked in this [excel](https://docs.google.com/spreadsheets/d/1ovGYoTcQZkRDXEwAZ5278RPfPh73KC0zgMktf7-vxNM/edit?gid=0#gid=0).

