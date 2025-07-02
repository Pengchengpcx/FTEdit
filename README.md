# Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing

Official Implementation of **CVPR 2025** for FTEdit: Tuning-free image editing based on Flow Transformer

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/pdf/2411.15843" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2411.10499-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://pengchengpcx.github.io/EditFT/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
</div>

**Pengcheng Xu | Boyuan Jiang | Xiaobin Hu | Donghao Luo | Qingdong He | Jiangning Zhang | Chengjie Wang | Yunsheng Wu | Charles Ling | Boyu Wang**

> Western University
> 
> Tencent


# Setup
Install the conda environment:
```bash
conda env create -f environment.yml
conda activate ftedit
```

# Image Editing

## Edit generated images
You can manipulate the images along with the generation process with AdaLN injection.

``` python
python edit_gensd35.py --inv_cfg 4.0 --recov_cfg 4.0 --skip_steps 0\
                       --src_prompt 'a silver shorthair cat sit on the wooden table'\
                       --tar_prompt 'a golden shorthair cat sit on the wooden table'\
                       --saved_path ./\
                       --seed 2024\
                       --model_path 'model path'
```


You can try different prompts to control different editing types. Here we provide some examples:

``` python
[
  ['three apples on a silver plate', 'two apples on a silver plate'],
  ['a beauty with the smilling face', 'a beauty with a sad face'],
  ['A monkey holding a sign reading ”Scaling diffusion model is awesome!', 'A monkey holding a sign reading ”Scaling transformers model is awesome!'],
  ['a cheetah standing on the wooden table', 'a cheetah sitting on the wooden table'],
]
```


## Consistent generation
AdaLN can also be used to generate images with consistent ID by maintaing the invaraint text prompt. Application on real images with ID masks can also be combined and explored.

``` python
[
  ['an origami seesaw on the table', 'an origami seesaw, playing football'],
  ['an origami seesaw on the table', 'an origami seesaw, surfing on the sea'],
  ['an origami seesaw on the table', 'an origami seesaw, sitting in the meadow'],
]
```




## Edit real images
We provide script to edit image for evaluation. We build the evaluation protocal based on the [PIE benchmark](https://github.com/cure-lab/PnPInversion). Apart from the proposed AdaLN invariance control mechanism, we also add the Attention injection mechanism. You can change the hyperparameters `--ly_ratio` for AdaLN and `--attn_ratio` for Attention to control the ratio of timesteps to copy features. We also integrate the skip-step option `--skip_steps` that skip the first steps during editing and inversion to control the inversion steps. This can also help to maintain the invaraince.


``` python
python edit_real_sd35.py --inv_cfg 1 --recov_cfg 2\
                         --skip_steps 0 --ly_ratio 1.0 --attn_ratio 0.15\
                         --src_path "path_to_dataset"\
                         --saved_path "path_to_edited_images"\
                         --model_path 'model path'


python evaluate.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image"  "clip_similarity_target_image" "clip_similarity_target_image_edit_part"\
 --result_path evaluation_cfg12_ly1.0.csv\
 --edit_category_list 0 1 2 3 4 5 6 7 8 9 --tar_image_folder "path_to_edited_images"\
 --tar_method "sd35_ftedit"\
 --src_image_folder "path_to_original_images"\
 --annotation_mapping_file "mapping_file.json"\
 --model_path 'model path'
```


The intuition of choosing these parameters is that you can generally set `--ly_ratio` as 1.0 if the inversion of the real image is accuarate and approximates the real generation process. In this case, the text-to-image alingment is not mismatched, and the AdaLN injection can flexibly control of image contents by manipulating text prompt. You can also add a small ratio of Attention injection `--attn_ratio` as 0.1 or 0.2. However, too much attention injection may hinder the editing effect since it inject both desired edited and non-target edited features.

You can also edit a single image with the following script:


``` python
python edit_real_sd35_singleimg.py --inv_cfg 1 --recov_cfg 2\
                         --skip_steps 0 --ly_ratio 1.0 --attn_ratio 0.15\
                         --src_path 'examples/1.jpg'\
                         --src_prompt 'a cup of coffee with drawing of tulip putted on the wooden table.'\
                         --tar_prompt 'a cup of coffee with drawing of lion putted on the wooden table.'\
                         --saved_path './'\
                         --model_path 'model_path'


python edit_real_sd35_singleimg.py --inv_cfg 1 --recov_cfg 2\
                         --skip_steps 0 --ly_ratio 1.0 --attn_ratio 0.1\
                         --src_path 'examples/2.jpg'\
                         --src_prompt 'A person with arms down.'\
                         --tar_prompt 'A person with arms crossed.'\
                         --saved_path './'\
                         --model_path 'model_path'


python edit_real_sd35_singleimg.py --inv_cfg 1 --recov_cfg 2\
                         --skip_steps 0 --ly_ratio 1.0 --attn_ratio 0.0\
                         --src_path 'examples/3.jpg'\
                         --src_prompt "a logo of a 'bird' shape in a black background."\
                         --tar_prompt "a logo of a 'X' shape in a black background."\
                         --saved_path './'\
                         --model_path 'model_path'
```


# TODO
Release the extended version on convergence analysis and curated data.




# Citation
If you think our work helpful, please cite our paper. Thanks for your interest and support!



``` bibtex
@inproceedings{xu2025unveil,
  title={Unveil inversion and invariance in flow transformer for versatile image editing},
  author={Xu, Pengcheng and Jiang, Boyuan and Hu, Xiaobin and Luo, Donghao and He, Qingdong and Zhang, Jiangning and Wang, Chengjie and Wu, Yunsheng and Ling, Charles and Wang, Boyu},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={28479--28489},
  year={2025}
}
```



