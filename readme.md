# Enhancing Virtual Shoe Try-On with Neural Rendering
## Rapid Inpainting - Engineering Master's Project

SOTA virtual shoe try-on methods lack a desired realism. If the user’s actual shoe is longer or wider than the virtual shoe, the user’s real shoes will be visible from underneath the AR shoes, thus leading to a poor user experience.

This is rectifiable via image inpainting. A segmentation model is used to calculate the errenous areas where the user's foot is visible from underneath the virtual shoes. An inpainting algorithm then fills in these areas with background. 

A prototype was delivered by using Mask Aware Transformer for image inpainting. However, this large model is quite slow, and therefore not suitable for real-time (30-60fps) performance on mobile devices.

This repository includes vast amounts of experimentation in developing a rapid inpainting network:
1. Custom Encoder-Decoder
2. UNet with MobileNetV2 encoder backbone
3. Segformer
4. MobileViT