import torch
from mmdit.sd35_pipeline import StableDiffusion3Pipeline
from inversion.flow_fixpoint_residual_new import Inversed_flow_fixpoint_residual
from inversion.inv_utils import fix_seed, load_PIE_images, view_images
from controller import attn_norm_ctrl_sd35
import numpy as np
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--skip_steps', type=int, default=0)
    parser.add_argument('--inv_cfg', type=float, default=1.0)
    parser.add_argument('--recov_cfg', type=float, default=1.0)
    parser.add_argument('--ly_ratio', type=float, default=1.0)
    parser.add_argument('--attn_ratio', type=float, default=1.0)
    parser.add_argument('--src_prompt', type=str, default="",)
    parser.add_argument('--tar_prompt', type=str, default="",)
    parser.add_argument('--src_path', type=str, default=None, required=True)
    parser.add_argument('--saved_path', type=str, default=None, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2024)
    return parser.parse_args()

    
if __name__ == "__main__":
    args = get_parser()
    fix_seed(args.seed)
    g = torch.Generator(device=args.device).manual_seed(args.seed)

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, \
                                                    torch_dtype=torch.bfloat16, local_files_only=True,)
    pipe = pipe.to(args.device)
    pipe.transformer.eval()
    pipe.vae.eval()   
    
    invf = Inversed_flow_fixpoint_residual(pipe, args.num_steps, args.device, args.inv_cfg, args.recov_cfg, args.skip_steps, args.saved_path)
        
    ######## Read images from the PIE
    ori_prp_list, edi_prp_list, img_list, edi_ins_list, bld_list, mask_list = load_PIE_images(args.src_path, edit_category_list=["0","1","2","3","4","5","6","7","8","9"])

    
    for i in range(len(ori_prp_list)):
        img_f = img_list[i]
        
        src_prompt = ori_prp_list[i].replace("[", "").replace("]", "")
        tar_prompt = edi_prp_list[i].replace("[", "").replace("]", "")
        
        print(src_prompt)
        print(tar_prompt)
        
        prompts = [src_prompt, tar_prompt]
        
        ################################################################### edit
        attn_norm_ctrl_sd35.register_attention_control_sd35(pipe, None, None)
        all_latents = invf.euler_flow_inversion(prompt=src_prompt, image=img_f,
                                                num_fixpoint_steps=2, 
                                                average_step_ranges=(0, 5), 
                                                )
        
        ###### edit with fixpoint + residual composation
        controller_ada = attn_norm_ctrl_sd35.Adalayernorm_replace(prompts, args.num_steps, args.ly_ratio, pipe.tokenizer, pipe.tokenizer_3, device="cuda")
        controller_attn = attn_norm_ctrl_sd35.SD3attentionreplace(prompts, args.num_steps, args.attn_ratio)
        attn_norm_ctrl_sd35.register_attention_control_sd35(pipe, controller_attn, controller_ada)
        image1, image2 = invf.edit_img_with_residual(prompts, all_latents, controller_ada)
        result_path = args.saved_path + '/' + img_f.split('/')[-1][:-4]
        image_list = [np.squeeze(x) for x in [image1, image2]]
        view_images(image_list, result_path)


