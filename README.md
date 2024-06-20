# MMSSL: Multi-Modal Self-Supervised Learning for Recommendation

PyTorch implementation for WWW 2023 paper [Multi-Modal Self-Supervised Learning for Recommendation](https://arxiv.org/pdf/2302.10632.pdf).

<p align="center">
<img src="./MMSSL.png" alt="MMSSL" />
</p>

MMSSL is a new multimedia recommender system which integrates the generative modality-aware collaborative self-augmentation and the contrastive cross-modality dependency encoding. It achieves better performance than existing SOTA multi-model recommenders.


<h2>Dependencies </h2>

* Python >= 3.9.13
* [Pytorch](https://pytorch.org/) >= 1.13.0+cu116
* [dgl-cuda11.6](https://www.dgl.ai/) >= 0.9.1post1




<h2>Usage </h2>

Start training and inference as:

```
cd MMSSL
python ./main.py --dataset {DATASET}
```
Supported datasets:  `Amazon-Baby`, `Amazon-Sports`, `Tiktok`, `Allrecipes`


<h2> Datasets </h2>

  ```
  ├─ MMSSL/ 
      ├── data/
        ├── tiktok/
        ...
  ```
  |    Dataset   |   |  Amazon  |      |   |          |      |   |  Tiktok  |     |     |   | Allrecipes |    |
|:------------:|:-:|:--------:|:----:|:-:|:--------:|:----:|:-:|:--------:|:---:|:---:|:-:|:----------:|:--:|
|   Modality   |   |     V    |   T  |   |     V    |   T  |   |     V    |  A  |  T  |   |      V     |  T |
|   Embed Dim  |   |   4096   | 1024 |   |   4096   | 1024 |   |    128   | 128 | 768 |   |    2048    | 20 |
|     User     |   |   35598  |      |   |   19445  |      |   |   9319   |     |     |   |    19805   |    |
|     Item     |   |   18357  |      |   |   7050   |      |   |   6710   |     |     |   |    10067   |    |
| Interactions |   |  256308  |      |   |  139110  |      |   |   59541  |     |     |   |    58922   |    |
|   Sparsity   |   | 99.961\% |      |   | 99.899\% |      |   | 99.904\% |     |     |   |  99.970\%  |    |

- `2024.3.20 baselines LLATTICE and MICRO uploaded`: 📢📢📢📢🌹🔥🔥🚀🚀 Because baselines `LATTICE` and `MICRO` require some minor modifications, we provide code that can be easily run by simply modifying the dataset path.
- `2023.11.1 new multi-modal datastes uploaded`: 📢📢🔥🔥🌹🌹🌹🌹 We provide new multi-modal datasets `Netflix` and `MovieLens`  (i.e., CF training data, multi-modal data including `item text` and `posters`) of new multi-modal work [LLMRec](https://github.com/HKUDS/LLMRec) on Google Drive. 🌹We hope to contribute to our community and facilitate your research~

- `2023.3.23 update(all datasets uploaded)`: We provide the processed data at [Google Drive](https://drive.google.com/drive/folders/1AB1RsnU-ETmubJgWLpJrXd8TjaK_eTp0?usp=share_link). 
- `2023.3.24 update`: The official website of the `Tiktok` dataset has been closed. Thus, we also provide many other versions of preprocessed [Tiktok](https://drive.google.com/drive/folders/1hLvoS7F0R_K0HBixuS_OVXw_WbBxnshF?usp=share_link).  We spent a lot of time pre-processing this dataset, so if you want to use our preprocessed Tiktok in your work please cite.

🚀🚀 The provided dataset is compatible with multi-modal recommender models such as [MMSSL](https://github.com/HKUDS/MMSSL), [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), and [MICRO](https://github.com/CRIPAC-DIG/MICRO) and requires no additional data preprocessing, including (1) basic user-item interactions and (2) multi-modal features.


<h3> Multi-modal Datasets </h3>
🌹🌹 Please cite our paper if you use the 'netflix' dataset~ ❤️  

We collected a multi-modal dataset using the original [Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) released on the [Kaggle](https://www.kaggle.com/) website. The data format is directly compatible with state-of-the-art multi-modal recommendation models like [LLMRec](https://github.com/HKUDS/LLMRec), [MMSSL](https://github.com/HKUDS/MMSSL), [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), [MICRO](https://github.com/CRIPAC-DIG/MICRO), and others, without requiring any additional data preprocessing.

 `Textual Modality:` We have released the item information curated from the original dataset in the "item_attribute.csv" file. Additionally, we have incorporated textual information enhanced by LLM into the "augmented_item_attribute_agg.csv" file. (The following three images represent (1) information about Netflix as described on the Kaggle website, (2) textual information from the original Netflix Prize Data, and (3) textual information augmented by LLMs.)
<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 10px;">
   <img src="./image/textual_data1.png" alt="Image 1" style="width:270px;height:180px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data' on Kaggle.</figcaption> -->
  </figure>

  <figure style="text-align: center; margin: 10px;">
    <img src="./image/textual_data2.png" alt="Image 2" style="width:270px;height:180px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data'.</figcaption> -->
  </figure>

  <figure style="text-align: center; margin: 10px;">
    <img src="./image/textual_data3.png" alt="Image 2" style="width:270px;height:180px;">
<!--     <figcaption>LLM-augmented textual data.</figcaption> -->
  </figure>  
</div>
 
 `Visual Modality:` We have released the visual information obtained from web crawling in the "Netflix_Posters" folder. (The following image displays the poster acquired by web crawling using item information from the Netflix Prize Data.)
 <div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 10px;">
   <img src="./image/visiual_data1.png" alt="Image 1" style="width:690px;height:590px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data' on Kaggle.</figcaption> -->
  </figure>
</div>
 

<h3> Original Multi-modal Datasets & Augmented Datasets </h3>
 <div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 10px;">
   <img src="./image/datasets.png" alt="Image 1" style="width:480px;height:270px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data' on Kaggle.</figcaption> -->
  </figure>
</div>


<br>
<p>

<h3> Download the Netflix dataset. </h3>
🚀🚀
We provide the processed data (i.e., CF training data & basic user-item interactions, original multi-modal data including images and text of items, encoded visual/textual features and LLM-augmented text/embeddings).  🌹 We hope to contribute to our community and facilitate your research 🚀🚀 ~

- `netflix`: [Google Drive Netflix](https://drive.google.com/drive/folders/1BGKm3nO4xzhyi_mpKJWcfxgi3sQ2j_Ec?usp=drive_link).  [🌟(Image&Text)](https://drive.google.com/file/d/1euAnMYD1JBPflx0M86O2M9OsbBSfrzPK/view?usp=drive_link)



<h3> Encoding the Multi-modal Content. </h3>

We use [CLIP-ViT](https://huggingface.co/openai/clip-vit-base-patch32) and [Sentence-BERT](https://www.sbert.net/) separately as encoders for visual side information and textual side information.



<h1> Citing </h1>

If you find this work helpful to your research, please kindly consider citing our paper.


```
@inproceedings{wei2023multi,
  title={Multi-Modal Self-Supervised Learning for Recommendation},
  author={Wei, Wei and Huang, Chao and Xia, Lianghao and Zhang, Chuxu},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={790--800},
  year={2023}
}
```
<!-- or -->

<!-- @inproceedings{wei2023multi,
  title={Multi-Modal Self-Supervised Learning for Recommendation},
  author={Wei, Wei and Huang, Chao and Xia, Lianghao and Zhang, Chuxu},
  booktitle={Proceedings of the Web Conference (WWW)},
  year={2023}
}
 -->


## Acknowledgement

The structure of this code is largely based on [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), [MICRO](https://github.com/CRIPAC-DIG/MICRO). Thank them for their work.

