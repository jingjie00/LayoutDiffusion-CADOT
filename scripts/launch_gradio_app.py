"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import functools

import torch
import torch as th
from omegaconf import OmegaConf

from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.util import fix_seed
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from layout_diffusion.dataset.data_loader import build_loaders
from scripts.get_gradio_demo import get_demo
from layout_diffusion.dataset.util import image_unnormalize_batch
import numpy as np

{'small-object': 1, 'basketball field': 2, 'building': 3, 'crosswalk': 4, 'football field': 5, 'graveyard': 6, 'large vehicle': 7, 'medium vehicle': 8, 'playground': 9, 'roundabout': 10, 'ship': 11, 'small vehicle': 12, 'swimming pool': 13, 'tennis court': 14, 'train': 15, '__image__': 0, '__null__': 16}


@torch.no_grad()
def layout_to_image_generation(cfg, model_fn, noise_schedule, custom_layout_dict):
    print(custom_layout_dict)

    layout_length = cfg.data.parameters.layout_length

    model_kwargs = {
        'obj_bbox': torch.zeros([1, layout_length, 4]),
        'obj_class': torch.zeros([1, layout_length]).long().fill_(object_name_to_idx['__null__']),
        'is_valid_obj': torch.zeros([1, layout_length])
    }
    model_kwargs['obj_class'][0][0] = object_name_to_idx['__image__']
    model_kwargs['obj_bbox'][0][0] = torch.FloatTensor([0, 0, 1, 1])
    model_kwargs['is_valid_obj'][0][0] = 1.0

    for obj_id in range(1, custom_layout_dict['num_obj']-1):
        obj_bbox = custom_layout_dict['obj_bbox'][obj_id]
        obj_class = custom_layout_dict['obj_class'][obj_id]
        if obj_class == 'pad':
            obj_class = '__null__'

        model_kwargs['obj_bbox'][0][obj_id] = torch.FloatTensor(obj_bbox)
        model_kwargs['obj_class'][0][obj_id] = object_name_to_idx[obj_class]
        model_kwargs['is_valid_obj'][0][obj_id] = 1

    print(model_kwargs)


    wrappered_model_fn = model_wrapper(
        model_fn,
        noise_schedule,
        is_cond_classifier=False,
        total_N=1000,
        model_kwargs=model_kwargs
    )
    for key in model_kwargs.keys():
        model_kwargs[key] = model_kwargs[key].cuda()

    dpm_solver = DPM_Solver(wrappered_model_fn, noise_schedule)

    x_T = th.randn((1, 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size)).cuda()

    sample = dpm_solver.sample(
        x_T,
        steps=int(cfg.sample.timestep_respacing[0]),
        eps=float(cfg.sample.eps),
        adaptive_step_size=cfg.sample.adaptive_step_size,
        fast_version=cfg.sample.fast_version,
        clip_denoised=False,
        rtol=cfg.sample.rtol
    )  # (B, 3, H, W), B=1

    sample = sample.clamp(-1, 1)

    generate_img = np.array(sample[0].cpu().permute(1,2,0) * 127.5 + 127.5, dtype=np.uint8)
    # generate_img = np.transpose(generate_img, (1,0,2))
    print(generate_img.shape)




    print("sampling complete")

    return generate_img


@torch.no_grad()
def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./configs/COCO-stuff_256x256/LayoutDiffusion-v7_small.yaml')
    parser.add_argument("--share", action='store_true')

    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    print(OmegaConf.to_yaml(cfg))

    print("creating model...")
    model = build_model(cfg)
    model.cuda()
    print(model)

    if cfg.sample.pretrained_model_path:
        print("loading model from {}".format(cfg.sample.pretrained_model_path))
        checkpoint = torch.load(cfg.sample.pretrained_model_path, map_location="cpu")

        try:
            model.load_state_dict(checkpoint, strict=True)
            print('successfully load the entire model')
        except:
            print('not successfully load the entire model, try to load part of model')

            model.load_state_dict(checkpoint, strict=False)

    model.cuda()
    if cfg.sample.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, **kwargs):
        assert obj_class is not None
        assert obj_bbox is not None

        cond_image, cond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj
        )
        cond_mean, cond_variance = th.chunk(cond_image, 2, dim=1)

        obj_class = th.ones_like(obj_class).fill_(model.layout_encoder.num_classes_for_layout_object - 1)
        obj_class[:, 0] = 0

        obj_bbox = th.zeros_like(obj_bbox)
        obj_bbox[:, 0] = th.FloatTensor([0, 0, 1, 1])

        is_valid_obj = th.zeros_like(obj_class)
        is_valid_obj[:, 0] = 1.0

        if obj_mask is not None:
            obj_mask = th.zeros_like(obj_mask)
            obj_mask[:, 0] = th.ones(obj_mask.shape[-2:])

        uncond_image, uncond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj
        )
        uncond_mean, uncond_variance = th.chunk(uncond_image, 2, dim=1)

        mean = cond_mean + cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

        if cfg.sample.sample_method in ['ddpm', 'ddim']:
            return [th.cat([mean, cond_variance], dim=1), cond_extra_outputs]
        else:
            return mean

    print("creating diffusion...")

    noise_schedule = NoiseScheduleVP(schedule='linear')

    print('sample method = {}'.format(cfg.sample.sample_method))
    print("sampling...")

    return cfg, model_fn, noise_schedule


if __name__ == "__main__":
    cfg, model_fn, noise_schedule = init()

    demo = get_demo(layout_to_image_generation, cfg, model_fn, noise_schedule)

    demo.launch(share=cfg.share)
