<p align="center">
  <img src="https://github.com/1ucky40nc3/medicus/raw/main/docs/medicus.png">
</p>

--------------------------------------------------------------------

This repository aims to implement methods to process medical images.
Theese images may be CT  or MRI scans that shall be inspected via the following tasks:
- [ ] classification
- [ ] segmentation
  - [ ] 2D
  - [ ] 3D
- [ ] generation

In order to conduct the experiments we also compile a [list of datasets](#datasets)
- [ ] Custom simulated datasets
  - [ ] Custom shape segmentation datasets
  - [ ] Custom medical segmentation datasets
- [ ] The Medical Segmentation Decathlon

[Our implementations](#methods) include methods from some papers such as:
- [ ] U-Net[^1]
- [ ] U^2-Net[^2]



# Datasets
- [Custom simulated datasets](https://drive.google.com/drive/folders/1t6fduzQQWWabikIKZY22b2ijtewbHEvR?usp=sharing)
- [The Medical Segmentation Decathlon](http://medicaldecathlon.com/)[^3]


# Methods

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](medicus/models/unet.py)[^1]
- [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](medicus/models/u2net.py)[^2]


# References
[^1]: [Ronneberger, O., Fischer, P., and Brox, T.. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.](https://arxiv.org/abs/1505.04597)
[^2]: [Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R. Zaiane, and Martin Jagersand 2020. U2-Net: Going deeper with nested U-structure for salient object detection. Pattern Recognition, 106, p.107404.](https://arxiv.org/abs/2005.09007)
[^3]: [
Antonelli, M., Reinke, A., Bakas, S., Farahani, K., AnnetteKopp-Schneider, Landman, B., Litjens, G., Menze, B., Ronneberger, O., Summers, R., Ginneken, B., Bilello, M., Bilic, P., Christ, P., Do, R., Gollub, M., Heckers, S., Huisman, H., Jarnagin, W., McHugo, M., Napel, S., Pernicka, J., Rhode, K., Tobon-Gomez, C., Vorontsov, E., Huisman, H., Meakin, J., Ourselin, S., Wiesenfarth, M., Arbelaez, P., Bae, B., Chen, S., Daza, L., Feng, J., He, B., Isensee, F., Ji, Y., Jia, F., Kim, N., Kim, I., Merhof, D., Pai, A., Park, B., Perslev, M., Rezaiifar, R., Rippel, O., Sarasua, I., Shen, W., Son, J., Wachinger, C., Wang, L., Wang, Y., Xia, Y., Xu, D., Xu, Z., Zheng, Y., Simpson, A., Maier-Hein, L., and Cardoso, M.. (2021). The Medical Segmentation Decathlon.](https://arxiv.org/abs/2106.05735)
