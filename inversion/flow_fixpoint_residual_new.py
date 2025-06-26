import torch
from mmdit.sd35_pipeline import StableDiffusion3Pipeline, retrieve_timesteps
from diffusers.image_processor import VaeImageProcessor
import numpy as np
from tqdm import tqdm
from PIL import Image



def load_1024(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((1024, 1024)))
    return image



'''
Rectified flow matching inversion with DDPM edit friendly inversion

First inverted with fixpoint and then when recover the image, calculate the mismatch of each state
add this into noise correction of each prediction.
'''
class Inversed_flow_fixpoint_residual:
    def __init__(self, model, steps, device, inv_cfg, recov_cfg, skip_steps, saved_path):
        self.model = model
        self.num_steps = steps
        self.device = device
        self.inv_cfg = inv_cfg
        self.recov_cfg = recov_cfg
        self.skip_steps = skip_steps
        self.saved_path = saved_path

    def get_embeddings(self, prompt):
        '''
        get the text embeddings for the model
        '''
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.model.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=True,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=self.device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )
        
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
        

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = (latents / self.model.vae.config.scaling_factor) + self.model.vae.config.shift_factor
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.float().cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.to(torch.bfloat16)
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = (latents - self.model.vae.config.shift_factor) * self.model.vae.config.scaling_factor
        return latents


    def encode_latent(self, image_path):
        '''
        encode the image latent
        '''
        image = load_1024(image_path)
        image_gt = self.image2latent(image)
        image_rec = self.latent2image(image_gt)

        image_rec = Image.fromarray(image_rec[0])
        image_name = image_path.split("/")[-1][:-4]

        return image_gt


    @torch.no_grad()
    def euler_flow_inversion(self, prompt, image, 
                             num_fixpoint_steps, average_step_ranges):
        '''
        invert rectified flow with 1st order euler method without correction
        xt = x_{t_1} - (sigma_t) * dx/dt
        '''
        # reverse the time step
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.get_embeddings(prompt)

        if self.inv_cfg > 0:
            self.prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            self.pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.model.scheduler, self.num_steps, self.device, None)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.model.scheduler.order, 0)
        self.model._num_timesteps = len(timesteps)

        # encode the image latent
        latent = self.encode_latent(image)

        all_latents = [latent.clone().detach()]

        # compute the predicted noise
        for i in tqdm(range(self.num_steps)):
            if i > self.num_steps - self.skip_steps -1:
                break
                
            t = timesteps[len(timesteps) - i -1]

            ### get better estimated z_t-1
            latent = self.fixpoint_step(latent, t, i, self.prompt_embeds, self.pooled_prompt_embeds, 
                                       num_fixpoint_steps, average_step_ranges)
            all_latents.append(latent.clone().detach())
            
        return all_latents
    
    
    @torch.no_grad()
    def recovery_img(self, prompt, latent, f_name, residual_list=None, return_latents=False):
        '''
        recovery the original image with the inversed latent
        '''
        noise_latent = latent.clone().detach()
        reconstruct_latents = [latent.clone().detach().float().cpu()]
        # noise_img = self.latent2image(noise_latent)
        # noise_img = Image.fromarray(noise_img[0])
        # noise_img.save(self.saved_path + '/{}_fm_noise.png'.format(f_name))
        # reverse the time step
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.get_embeddings(prompt)

        if self.recov_cfg > 0:
            all_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            all_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.model.scheduler, self.num_steps, self.device, None)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.model.scheduler.order, 0)
        self.model._num_timesteps = len(timesteps)    
        
        # compute the predicted noise
        for i in tqdm(range(self.num_steps)):
            if i < self.skip_steps:
                continue
                
            t = timesteps[i]   

            if self.recov_cfg > 0:
                latent_model_input = torch.cat([latent] * 2)
            else:
                latent_model_input = latent
            
            timestep = t.expand(latent_model_input.shape[0])
            if residual_list != None:
                residual = residual_list[i - self.skip_steps] 
            noise_pred = self.model.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=all_prompt_embeds,
                pooled_projections=all_pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            if self.recov_cfg > 0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.recov_cfg * (noise_pred_text - noise_pred_uncond)

            # inversion with euler method
            sample = latent.to(torch.float64)
            sigma = self.model.scheduler.sigmas[i]
            sigma_next = self.model.scheduler.sigmas[i + 1]
            
            if residual_list!=None:
                prev_sample = sample + (sigma_next - sigma) * noise_pred + residual
            else:
                prev_sample = sample + (sigma_next - sigma) * noise_pred
            
            reconstruct_latents.append(prev_sample.clone().float().detach().cpu())
            latent = prev_sample.to(noise_pred.dtype)

        rec_img = self.latent2image(latent) 
        
        rec_img = Image.fromarray(rec_img[0])
        if residual_list != None:
            rec_img.save(self.saved_path + '/{}_fixpoint_residual_recov.png'.format(f_name))
        else:
            rec_img.save(self.saved_path + '/{}_fixpoint_recov.png'.format(f_name))

        self.model.maybe_free_model_hooks()

        if return_latents:
            return reconstruct_latents

        
        
    @torch.no_grad()
    def fixpoint_step(self, latent, t, cur_step, prompt_embeds, pooled_prompt_embeds, \
    num_fixpoint_steps, average_step_ranges, ave_noise=True):
        '''
        fixpoint all steps over the timestep
        
        fixpoint + average noise
        '''
        avg_range = average_step_ranges

        nosie_pred_avg = None
        approximated_z_tp1 = latent.clone()

        for i in range(num_fixpoint_steps + 1):
            with torch.no_grad():
                prompt_embeds_in = prompt_embeds
                pooled_prompt_embeds_in = pooled_prompt_embeds
            
            # esitemate the noise with approximated z_tp
            if self.inv_cfg > 0:
                latent_model_input = torch.cat([approximated_z_tp1] * 2)
            else:
                latent_model_input = approximated_z_tp1
            
            timestep = t.expand(latent_model_input.shape[0])
            
            noise_pred = self.model.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds_in,
                pooled_projections=pooled_prompt_embeds_in,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            if self.inv_cfg > 0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.inv_cfg * (noise_pred_text - noise_pred_uncond)

            # calculate average noise
            if  i >= avg_range[0] and i < avg_range[1]:
                j = i - avg_range[0]
                if j == 0:
                    nosie_pred_avg = noise_pred.clone()
                else:
                    nosie_pred_avg = (j * nosie_pred_avg + noise_pred ) / (j + 1)
            
            # use Euler to invert current latent to the next latent
            sample = latent.to(torch.float32)
            sigma = self.model.scheduler.sigmas[self.num_steps -cur_step]
            sigma_next = self.model.scheduler.sigmas[self.num_steps -cur_step -1]
            approximated_z_tp1 = sample + (sigma_next - sigma) * noise_pred
            approximated_z_tp1 = approximated_z_tp1.to(noise_pred.dtype)

        # use average noisy latent to calculate the approximate latent
        if ave_noise and nosie_pred_avg != None:
            sample = latent.to(torch.float32)
            approximated_z_tp1 = sample + (sigma_next - sigma) * nosie_pred_avg
            approximated_z_tp1 = approximated_z_tp1.to(noise_pred.dtype)

        return approximated_z_tp1


    @torch.no_grad()
    def edit_img_with_residual(self, prompt, all_latents, controller):
        '''
        recovery the original image with the inversed latent and supplementary rsidual of each step
        '''
        latent = torch.cat([all_latents[-1].clone().detach()]*2, dim=0).to(self.device)
        
        residual_list = []
        
        # reverse the time step
        src_prompt_embeds, src_negative_prompt_embeds, src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds = self.get_embeddings(prompt[0])
        tar_prompt_embeds, tar_negative_prompt_embeds, tar_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds = self.get_embeddings(prompt[1])
        
        prompt_embeds = torch.cat([src_prompt_embeds, tar_prompt_embeds], dim=0)
        negative_prompt_embeds = torch.cat([src_negative_prompt_embeds, tar_negative_prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([src_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
        negative_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds], dim=0)
        
        if self.recov_cfg > 0:
            all_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            all_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.model.scheduler, self.num_steps, self.device, None)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.model.scheduler.order, 0)
        self.model._num_timesteps = len(timesteps)

        # compute the predicted noise
        for i in tqdm(range(self.num_steps)):
            if i < self.skip_steps:
                if controller is not None:
                    controller.cur_step += 1
                continue
                
            t = timesteps[i]

            if self.recov_cfg > 0:
                latent_model_input = torch.cat([latent] * 2)
            else:
                latent_model_input = latent
            
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.model.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=all_prompt_embeds,
                pooled_projections=all_pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            if self.recov_cfg > 0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.recov_cfg * (noise_pred_text - noise_pred_uncond)

            # inversion with euler method
            sample = latent.to(torch.float32)
            sigma = self.model.scheduler.sigmas[i]
            sigma_next = self.model.scheduler.sigmas[i + 1]
            
            prev_sample = sample + (sigma_next - sigma) * noise_pred
            src_prev_sample = prev_sample[0,:].clone().detach()
            # calculate the supplementary residual based on the source inversion
            residual = all_latents[-2 - (i - self.skip_steps)] - src_prev_sample
            residual_list.append(residual)
            
            # supplement
            prev_sample[0,:] += residual.squeeze(0)

            latent = prev_sample.to(noise_pred.dtype)
        
        image1 = self.latent2image(latent[0].unsqueeze(0))
        image2 = self.latent2image(latent[1].unsqueeze(0))

        self.model.maybe_free_model_hooks()
            
        return image1, image2


    @torch.no_grad()
    def edit_img(self, prompt, latent, controller):
        '''
        recovery the original image with the inversed latent
        '''
        latent = torch.cat([latent.clone().detach()]*2, dim=0).to(self.device)
        
        # reverse the time step
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.get_embeddings(prompt)

        if self.recov_cfg > 0:
            self.prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            self.pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.model.scheduler, self.num_steps, self.device, None)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.model.scheduler.order, 0)
        self.model._num_timesteps = len(timesteps)

        # compute the predicted noise
        for i in tqdm(range(self.num_steps)):
            if i < self.skip_steps:
                if controller is not None:
                    controller.cur_step += 1
                continue
                
            t = timesteps[i]

            if self.recov_cfg > 0:
                latent_model_input = torch.cat([latent] * 2)
            else:
                latent_model_input = latent
            
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.model.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=self.prompt_embeds,
                pooled_projections=self.pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            if self.recov_cfg > 0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.recov_cfg * (noise_pred_text - noise_pred_uncond)

            # inversion with euler method
            sample = latent.to(torch.float32)
            sigma = self.model.scheduler.sigmas[i]
            sigma_next = self.model.scheduler.sigmas[i + 1]

            prev_sample = sample + (sigma_next - sigma) * noise_pred
            latent = prev_sample.to(noise_pred.dtype)
        
        rec_img = self.latent2image(latent)

        self.model.maybe_free_model_hooks()
            
        return rec_img
    
