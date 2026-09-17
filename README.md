# BAFLE-DCT: Frequency-Selective Embedding in the DCT Domain (Published in WACV 2026)

A Saliency-Aware FNN slection-Based Steganography Framework for Imperceptible and Robust Data Embedding. 

## Project Overview

BAFLE-DCT leverages Deep Learning, Frequency-Domain Embedding, and Saliency Mapping to hide data inside images, making detection hard for both humans and deep-learning steganalysis models like SRNet and YeNet.

> **BAFLE-DCT: Bypassing Adversarial Filters via Frequency-Selective 
> Embedding in the DCT Domain**
> Thilina Mendis, Farah Kandah, Sathyanarayanan N. Aakur
> IEEE/CVF Winter Conference on Applications of Computer Vision
> (WACV), 2026

[![IEEE Paper](https://img.shields.io/badge/IEEE%20Paper-WACV%202026-blue?logo=ieee)](https://doi.org/10.1109/WACV61042.2026.00577)

**An extended version of this work is currently under peer review at an IEEE journal.**

---

## Dataset

A publicly accessible frequency-domain dataset of color images for teganography and steganalysis research which was take from a subset of imagenet dataset. It contains paired cover and stego images with varied payload sizes, embedding parameters, image categories, and ground-truth labels with per-image metadata.

### Contents

- **Image generation:** For the steganalysis evaluation, we organized the image pairs into a standard dataset structure with 5,761 cover & stego image pairs (70%) for training, 1,234 pairs (15%) for validation, and 1,235 pairs (15%) for testing.
- **Stego generation:** BAFLE-DCT embedding at payloads of
  100, 200, 300, 400, and 500 bits
- **Format:** [JPEG / PNG]
- **Categories:** [IMAGENET]

### Download

The public dataset (train + validation) is archived with a DOI on
Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22776317.svg)](https://doi.org/10.5281/zenodo.22776317) 

### Test set access

The test split is held out to support fair benchmarking and comparison across methods. Researchers may request access by emailing **btm0050@auburn.edu** with their name, affiliation, and a brief description of intended use.

### Intended users

Researchers and developers in steganography, steganalysis, secure multimedia systems, digital image forensics, and adversarial machine learning — for benchmarking detection models, comparing embedding
methods, and reproducing published results.

---

## License

- **Code:** MIT License (see `LICENSE`)
- **Dataset:** CC BY 4.0 (see `LICENSE-DATA`)

## Citation

If you use this code or the dataset, please cite the WACV 2026 paper:

```bibtex
@INPROCEEDINGS{11492105,
  author={Mendis, Thilina and Kandah, Farah and Aakur, Sathyanarayanan N.},
  booktitle={2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
  title={BAFLE-DCT: Bypassing Adversarial Filters via Frequency-Selective Embedding in the DCT Domain}, 
  year={2026},
  volume={},
  number={},
  pages={5967-5976},
  keywords={Payloads;Low earth orbit satellites;Military aircraft;Space technology;Filtering;Filters;Circuits and systems;Pixel;Digital images;Protocols;steganography;steganalysis;frequency-selective embedding;image security;deep learning;dct domain},
  doi={10.1109/WACV61042.2026.00577}
  }
```

## Contact

**Thilina Mendis**  
Computer Science and Software Engineering  
Auburn University  
📧 [btm0050@auburn.edu](mailto:btm0050@auburn.edu)  
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--0569--6690-a6ce39?logo=orcid&logoColor=white)](https://orcid.org/0009-0000-0569-6690)