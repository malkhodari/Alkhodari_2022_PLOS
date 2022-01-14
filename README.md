# Detection of COVID-19 in smartphone-based breathing recordings: a pre-screening deep learning tool

This repository includes all codes needed to reproduce the work presented at "Alkhodari M, Khandoker AH (2022) Detection of COVID-19 in smartphone-based breathing recordings: A pre-screening deep learning tool. PLOS ONE 17(1): e0262448. https://doi.org/10.1371/journal.pone.0262448". 

The "main_code" includes the following,
1) Loading shallow or deep breathing datasets (in .MAT format) - all resampled to 4,000 Hz (instead of the original 48,000.
2) Selecting only 64,000 samples (16 seconds) - if the signal is less than 16 seconds, it is padded with zeros.
3) Preparing COVID-19 and healthy subjects' dataset.
4) Extraction of hand-crafted features
5) Training of the CNN-BiLSTM neural network for the extraction of deep-activated features.
6) Updating the parameters of the network with the best 20 hand-crafted features alongside age and sex.

We also provide the full clinical profiles of patients and the short profiles (only age and sex) in .MAT format.

For any questions, please do not hesitate to contact the corresponding author.
