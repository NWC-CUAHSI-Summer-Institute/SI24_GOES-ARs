# Identifying Atmospheric Rivers on the West Coast of the United States with Geostationary Operational Environmental Satellite (GOES) Imagery

# Summary
This repository contains all necessary information (rationale, methodology, data, codes and results) used in identifying atmospheric rivers (ARs) in the Western United States.
The primary objective of this project is to explore the potential of GOES satellite data in accurately identifying ARs along the Pacific coast of North America. We developed and tested two machine learning models—a U-Net model and a random forest model—to identify ARs from GOES satellite imagery spanning the period from 2018 to 2020. 
The research questions guiding this investigation include: 
- Can optical satellite data from GOES, with its unique spatio-temporal resolution, be effectively utilized for the detection of ARs? Can this potential be extended to real-time monitoring capabilities?
- How accurately can U-Net and random forest models identify atmospheric rivers from GOES satellite imagery? Which of the two models performs best regarding accuracy, precision, recall, and other relevant metrics for AR identification? 

# Study Area
Our research focuses on the Pacific West Coast of North America, as illustrated in Figures 1 (a) and 1 (b). The study area encompasses a broad region from the state of Washington in the north to California in the south, extending westward over the Pacific Ocean. 
![image](https://github.com/user-attachments/assets/72728d6d-d991-4170-84d9-74eee1a8c689)

# Dataset
- GOES Imagery
The GOES series is operated by the National Oceanic and Atmospheric Administration (NOAA) of the United States. This study utilizes data from the GOES-R series, specifically employing the Advanced Baseline Imager (ABI) with 16 channels, offering a spatial resolution of 2 km for most channels and a temporal resolution of 15 minutes for full disk scans. Data retrieval for the GOES-17 and GOES-18 satellites from 2016 to present is facilitated by the GOES-2-go Python package (Version 2022.07.15), which accesses data through Amazon Web Services (AWS) as part of NOAA's Open Data Dissemination Program [GOES2Go](https://github.com/blaylockbk/goes2go). package in github.The raw data from the GOES satellites required conversion into a standardized projection format. This step involved transforming the native GOES projection into a rectilinear latitude-longitude projection to ensure consistency with the analysis requirements and facilitate accurate spatial referencing. Additionally, the GOES images were resized from their original resolution of 4500x4996 pixels to 512x512 pixels to facilitate machine learning training and improve computational efficiency.
- Labeled data
The dataset consists of continuous labeled data from 1979 to 2022, recorded at 6-hour intervals. This extensive temporal coverage provides a comprehensive view of the AR events. The data was sourced from [Rhodes](https://portal.nersc.gov/archive/home/a/arhoades/Shared/www/TE_ERA5_ARs). which utilized the advanced TempestExtremes v2.1 algorithm to detect these features. The paper associated can be found [here](https://gmd.copernicus.org/articles/14/5023/2021/). This algorithm is known for its robust capability in identifying and characterizing AR events, ensuring high-quality and reliable data for research and analysis.
- ERA5 climate reanalysis product
The retrieved data from ERA5 include wind components (u, v) and specific humidity (q) at multiple pressure levels (300 hPa to 1000 hPa). These data were used to calculate Integrated Water Vapor (IWV) and Integrated Vapor Transport (IVT). IWV represents the total column of water vapor by integrating specific humidity over the pressure levels, while IVT quantifies the horizontal transport of water vapor using specific humidity and wind components across these levels. We utilized ERA-5 reanalysis data to calculate IVT and IWV and developed a thresholding algorithm to identify AR objects. This algorithm applies specific thresholds for IVT and IWV and evaluates the minimum length and width of potential AR regions to meet typical AR criteria. The algorithm starts by applying threshold values for IVT and IWV to create an initial mask of potential AR regions. Regions meeting these thresholds are marked. Minimum length and width requirements for ARs are converted from kilometers to grid points, ensuring precise identification of contiguous AR regions. Each connected region in the initial mask is labeled and evaluated against these criteria, resulting in a robust final binary mask representing AR regions. This mask is used for further analysis and benchmarking, providing a reliable method for identifying ARs in the ERA-5 reanalysis dataset.

# Models
- CNN architecture

- 
# How to run?
- Data should be in this directory: ``

# Codes
- Data processing
- Code to access training and clippling data to Western US can be found here.
- Code to access selected GOES data can be found hereto Western US can be found here.
- Machine learning

# Report
- Report can be found here. (To be included)
- Supplemantary material can be found here. (To be included)
  
# References
Reviewed literature is tracked in this [excel](https://docs.google.com/spreadsheets/d/1ovGYoTcQZkRDXEwAZ5278RPfPh73KC0zgMktf7-vxNM/edit?gid=0#gid=0).

