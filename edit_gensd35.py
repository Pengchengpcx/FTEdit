import torch
from mmdit.sd35_pipeline import StableDiffusion3Pipeline
from inversion.flow_fixpoint_residual_new import Inversed_flow_fixpoint_residual
from inversion.inv_utils import fix_seed, view_images
from controller import attn_norm_ctrl_sd35
import numpy as np
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--skip_steps', type=int, default=7)
    parser.add_argument('--inv_cfg', type=float, default=1.0)
    parser.add_argument('--recov_cfg', type=float, default=1.0)
    parser.add_argument('--ly_ratio', type=float, default=1.0)
    parser.add_argument('--src_prompt', type=str, default="",)
    parser.add_argument('--tar_prompt', type=str, default="",)
    parser.add_argument('--saved_path', type=str, default=None, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=222)
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_parser()
    fix_seed(args.seed)
    g = torch.Generator(device=args.device).manual_seed(args.seed)

    ######### SD3 init
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, \
                                                    torch_dtype=torch.bfloat16, local_files_only=True,)
    pipe = pipe.to(args.device)
    invf = Inversed_flow_fixpoint_residual(pipe, args.num_steps, args.device, args.inv_cfg, args.recov_cfg, args.skip_steps, args.saved_path)

    prompts = [
            [args.src_prompt, args.tar_prompt],
            ]

    prompts = [
    ['an origami hedgehog', 'an origami hedgehog, playing the football'],
    ['an origami hedgehog', 'an origami hedgehog, surfing on the sea'],
    ['an origami hedgehog', 'an origami seeshedgehog, sitting in the meadow'],
    ]
    

    for i in range(len(prompts)):
        latents = torch.randn([1, 16, 128, 128], generator=g, device="cuda", dtype=torch.bfloat16, layout=None).to("cuda")
        
        ###### remove the T5 embedding from edit
        controller_ada = attn_norm_ctrl_sd35.Adalayernorm_replace(prompts[i], args.num_steps, [0.0, 0.8], invf.model.tokenizer, invf.model.tokenizer_3, device="cuda")
        controller_attn = attn_norm_ctrl_sd35.SD3attentionreplace(prompts[i], args.num_steps, [0.0, 0.1])
        
        attn_norm_ctrl_sd35.register_attention_control_sd35(invf.model, None, controller_ada)
        
        edited_latents = invf.edit_img(prompts[i], latents, controller_ada)
        image1, image2 = np.split(edited_latents, 2)
        result_path = args.saved_path + '/file_{}'.format(i)
        image_list = [np.squeeze(x) for x in [image1, image2]]
        view_images(image_list, result_path)
        
        
'''
python edit_gensd35.py --inv_cfg 4.0 --recov_cfg 4.0 --skip_steps 0\
                            --src_prompt 'a silver shorthair cat sit on the wooden table'\
                            --tar_prompt 'a golden shorthair cat sit on the wooden table'\
                            --saved_path ./\
                            --seed 2024\
                            --model_path /home/tione/notebook/byronjiang/ckpt/stable-diffusion-3.5-large
'''


