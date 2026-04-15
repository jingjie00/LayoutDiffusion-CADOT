import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.util import fix_seed

# Dictionary mapping object names to indices
object_name_to_idx = {"__image__": 0, "__null__": 1, "small-object": 2, "basketball field": 3, "building": 4, "crosswalk": 5, "football field": 6, "graveyard": 7, "large vehicle": 8, "medium vehicle": 9, "playground": 10, "roundabout": 11, "ship": 12, "small vehicle": 13, "swimming pool": 14, "tennis court": 15, "train": 16}



class LayoutImageGenerator:
    def __init__(self, config_path, model_path, seed=None):
        """
        Initialize the layout-to-image generator
        
        Args:
            config_path: Path to configuration YAML file
            model_path: Path to pretrained model checkpoint
            seed: Random seed for reproducibility
        """
        if seed is not None:
            fix_seed(seed)
            
        self.cfg = OmegaConf.load(config_path)
        self.cfg.sample.pretrained_model_path = model_path
        self.cfg.sample.classifier_free_scale = 1.0
        self.cfg.sample.timestep_respacing = ["25"]  # Default steps
        self.cfg.sample.sample_method = 'dpm_solver'
        
        print("Creating model...")
        self.model = build_model(self.cfg)
        self.model.cuda()
        
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        try:
            self.model.load_state_dict(checkpoint, strict=True)
            print('Successfully loaded the entire model')
        except:
            print('Could not load the entire model, loading partial model')
            self.model.load_state_dict(checkpoint, strict=False)
            
        self.model.cuda()
        if self.cfg.sample.use_fp16:
            self.model.convert_to_fp16()
        self.model.eval()
        
        # Setup noise schedule
        self.noise_schedule = NoiseScheduleVP(schedule='linear')
        
    def _create_model_fn(self):
        """Create the model function for the diffusion sampler"""
        def model_fn(x, t, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, **kwargs):
            assert obj_class is not None
            assert obj_bbox is not None

            cond_image, cond_extra_outputs = self.model(
                x, t,
                obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
                is_valid_obj=is_valid_obj
            )
            cond_mean, cond_variance = torch.chunk(cond_image, 2, dim=1)

            obj_class = torch.ones_like(obj_class).fill_(self.model.layout_encoder.num_classes_for_layout_object - 1)
            obj_class[:, 0] = 0

            obj_bbox = torch.zeros_like(obj_bbox)
            obj_bbox[:, 0] = torch.FloatTensor([0, 0, 1, 1])

            is_valid_obj = torch.zeros_like(obj_class)
            is_valid_obj[:, 0] = 1.0

            if obj_mask is not None:
                obj_mask = torch.zeros_like(obj_mask)
                obj_mask[:, 0] = torch.ones(obj_mask.shape[-2:])

            uncond_image, uncond_extra_outputs = self.model(
                x, t,
                obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
                is_valid_obj=is_valid_obj
            )
            uncond_mean, uncond_variance = torch.chunk(uncond_image, 2, dim=1)

            mean = cond_mean + self.cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

            if self.cfg.sample.sample_method in ['ddpm', 'ddim']:
                return [torch.cat([mean, cond_variance], dim=1), cond_extra_outputs]
            else:
                return mean
                
        return model_fn
    
    @torch.no_grad()
    def generate(self, layout, steps=25, classifier_free_scale=1.0, output_path=None):
        """
        Generate an image from a layout
        
        Args:
            layout: List of [obj_class, x0, y0, x1, y1] for each object
                   First object should always be 'image' with bbox [0,0,1,1]
            steps: Number of diffusion steps
            classifier_free_scale: Scale for classifier-free guidance
            output_path: Path to save the generated image (optional)
            
        Returns:
            Generated image as a numpy array (H, W, 3) in RGB format
        """
        self.cfg.sample.timestep_respacing = [str(steps)]
        self.cfg.sample.classifier_free_scale = classifier_free_scale
        
        # Prepare layout data
        layout_length = self.cfg.data.parameters.layout_length
        
        model_kwargs = {
            'obj_bbox': torch.zeros([1, layout_length, 4]),
            'obj_class': torch.zeros([1, layout_length]).long().fill_(object_name_to_idx['__null__']),
            'is_valid_obj': torch.zeros([1, layout_length])
        }
        
        # Set image as the first object
        model_kwargs['obj_class'][0][0] = object_name_to_idx['__image__']
        model_kwargs['obj_bbox'][0][0] = torch.FloatTensor([0, 0, 1, 1])
        model_kwargs['is_valid_obj'][0][0] = 1.0
        
        # Add other objects from the layout
        for obj_id, obj_data in enumerate(layout[1:], start=1):
            if obj_id >= layout_length:
                break
                
            obj_class, x0, y0, x1, y1 = obj_data
            
            if obj_class == 'pad':
                obj_class = '__null__'
        
            model_kwargs['obj_bbox'][0][obj_id] = torch.FloatTensor([x0, y0, x1, y1])
            model_kwargs['obj_class'][0][obj_id] = object_name_to_idx[obj_class]
            model_kwargs['is_valid_obj'][0][obj_id] = 1
        
        # Create model function and wrap it
        model_fn = self._create_model_fn()
        wrappered_model_fn = model_wrapper(
            model_fn,
            self.noise_schedule,
            is_cond_classifier=False,
            total_N=1000,
            model_kwargs=model_kwargs
        )
        
        # Move tensors to GPU
        for key in model_kwargs.keys():
            model_kwargs[key] = model_kwargs[key].cuda()
            
        # Setup DPM solver
        dpm_solver = DPM_Solver(wrappered_model_fn, self.noise_schedule)
        
        # Generate random noise
        x_T = torch.randn((1, 3, self.cfg.data.parameters.image_size, self.cfg.data.parameters.image_size)).cuda()
        
        # Sample from the model
        sample = dpm_solver.sample(
            x_T,
            steps=steps,
            eps=float(self.cfg.sample.eps),
            adaptive_step_size=self.cfg.sample.adaptive_step_size,
            fast_version=self.cfg.sample.fast_version,
            clip_denoised=False,
            rtol=self.cfg.sample.rtol
        )
        
        # Process the generated image
        sample = sample.clamp(-1, 1)
        generate_img = np.array(sample[0].cpu().permute(1,2,0) * 127.5 + 127.5, dtype=np.uint8)
        
        # Save the images if output path is provided
        if output_path:
            # Save generated image
            Image.fromarray(generate_img).save(output_path)
            print(f"Generated image saved to {output_path}")
            
            # Save original image (x.png)
            original_path = output_path.replace('.png', '_x.png')
            Image.fromarray(generate_img).save(original_path)
            print(f"Original image saved to {original_path}")
            
        return generate_img

if __name__ == "__main__":
    #/home/jtan/LayoutDiffusion-CADOT/pretrained_models/COCO-stuff_256x256_LayoutDiffusion_small_ema_1700000.pt
    parser = argparse.ArgumentParser(description="Generate image from layout")
    parser.add_argument("--config_file", type=str, default='./configs/COCO-stuff_256x256/LayoutDiffusion_cadot.yaml')
    parser.add_argument("--model_path", type=str, help="Path to pretrained model", default='./log/COCO-stuff_256x256/LayoutDiffusion_cadot/ema_0.9999_0230000.pt')

    parser.add_argument("--output_path", type=str, default="generated_image.png", help="Path to save generated image")
    parser.add_argument("--steps", type=int, default=25, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=2333, help="Random seed")
    args = parser.parse_args()
    
    # Initialize the generator
    generator = LayoutImageGenerator(args.config_file, args.model_path, args.seed)
    
    # Example layout: [class, x0, y0, x1, y1]
    # First object must be 'image' with coordinates [0,0,1,1]
    example_layout = [['image', 0.0, 0.0, 1.0, 1.0],
                     ['playground', 0.0, 0.5791666507720947, 1.0, 1.0],
                     ['building', 0.001687500043772161, 0.41574999690055847, 0.3943749964237213, 0.9820416569709778],
                     ['crosswalk', 0.0, 0.0, 0.25312501192092896, 0.4791666567325592]]

    # Generate the image
    generated_image = generator.generate(
        example_layout, 
        steps=args.steps,
        output_path=args.output_path
    )
    
    print("Image generation complete!")