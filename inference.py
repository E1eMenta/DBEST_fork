import einops
from PIL import Image
import cv2
from loguru import logger
import torch
import numpy as np

from t2tldm.model_256 import ldm, tokenizer, scheduler, PNDM_scheduler, MAX_NUM_WORDS, NUM_DDIM_STEPS
from t2tldm.utils import img_utils, prompt_utils, denoising_utils
from t2tldm.baselines.imagic.method import run_finetune_et_model
from t2tldm.methods.optimization import opt_emb_DDPM_text_recog


class DBESTTextModulator:
    """Text Modulation Model with DBEST"""

    def __init__(
        self,
        pretrained_weight_path="ft_et_text_syntext4chars_100k.ckpt",
        yaml_file="train_abinet.yaml",
        checkpoint_tr="best-train-abinet.pth",
        device="cpu",
        seed=888,
    ):
        """
        Initialize the DBEST text modulation model.

        Args:
            pretrained_weight_path: Path to the pretrained weights
            yaml_file: Path to the YAML configuration file
            checkpoint_tr: Path to the checkpoint for text recognition
            device: Computation device
            seed: Random seed for reproducibility
        """
        # Store configuration
        self.device = device
        self.seed = seed
        self.yaml_file = yaml_file
        self.checkpoint_tr = checkpoint_tr
        self.a_prompt = "A text that reads: "

        # Load model and weights
        logger.info(f"Initializing model on {device}")
        self.ldm = ldm
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.PNDM_scheduler = PNDM_scheduler

        # Load weights
        logger.info(f"Loading weights from {pretrained_weight_path}")
        et_weight_syntext = torch.load(pretrained_weight_path, map_location=torch.device("cpu"))
        self.ldm.unet.load_state_dict(et_weight_syntext)

        # Set up generator
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        logger.success("Model initialized successfully")

    def infer(
        self,
        input_image,
        src_prompt,
        tgt_prompt,
        guidance_scale=0.3,
        use_ddim=True,
        target_size=256,
        finetune_iterations=1500,
        optimization_iterations=1000,
    ):
        """
        Run inference on the input image to modulate text.

        Args:
            input_image: Input RGB numpy uint8 image
            src_prompt: Source text
            tgt_prompt: Target text
            guidance_scale: Guidance scale for sampling (0.1-1.0)
            use_ddim: Whether to use DDIM (True) or PNDM (False) sampler
            target_size: Target image size
            finetune_iterations: Number of iterations for finetuning
            optimization_iterations: Number of iterations for optimization

        Returns:
            RGB numpy uint8 image with modulated text
        """
        logger.info(f"Processing image with source text: '{src_prompt}' -> target text: '{tgt_prompt}'")

        # Preprocess image
        img = Image.fromarray(np.uint8(input_image))
        img = img.convert("RGB")
        img = img.resize((target_size, target_size))

        # Prepare prompts
        src_p = [self.a_prompt + f'"{src_prompt}"']
        tgt_p = [self.a_prompt + f'"{tgt_prompt}"']

        # Generate embeddings for source prompt
        src_uncond_emb, src_cond_emb = prompt_utils.gen_init_prompt_to_emb(
            self.ldm, self.tokenizer, src_p, MAX_NUM_WORDS, device=self.device
        )
        src_context = torch.cat([src_uncond_emb, src_cond_emb])

        # Prepare input
        src_z0, src_zT = img_utils.prepare_input_gradio(
            self.ldm, self.scheduler, src_context, img, self.generator, NUM_DDIM_STEPS, target_size, device=self.device
        )

        # Finetune model
        logger.info(f"Finetuning model for {finetune_iterations} iterations")
        run_finetune_et_model(
            self.ldm,
            self.ldm.scheduler,
            src_z0.detach(),
            src_cond_emb.detach(),
            num_iter=finetune_iterations,
            device=self.device,
        )

        # Generate embeddings for target prompt
        tgt_uncond_emb, tgt_cond_emb = prompt_utils.gen_init_prompt_to_emb(
            self.ldm, self.tokenizer, tgt_p, MAX_NUM_WORDS, device=self.device
        )

        # Optimize embeddings
        logger.info(f"Optimizing embeddings for {optimization_iterations} iterations")
        opt_tgt_emb = opt_emb_DDPM_text_recog(
            self.ldm,
            self.ldm.scheduler,
            src_z0,
            src_cond_emb,
            tgt_cond_emb,
            target_text=tgt_prompt.lower(),
            yaml_file=self.yaml_file,
            checkpoints_path=self.checkpoint_tr,
            num_iter=optimization_iterations,
            save_dir="",
            device=self.device,
        )

        # Convert source latent to image
        src_x0 = img_utils.latent2im(self.ldm, src_z0)
        src_x0 = (
            (einops.rearrange(src_x0.detach(), "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )
        src_x0 = src_x0[0]

        # Prepare context for sampling
        context = torch.cat([tgt_uncond_emb, opt_tgt_emb.detach()])

        # Run diffusion sampling
        logger.info(f"Running {'DDIM' if use_ddim else 'PNDM'} sampling with guidance scale {guidance_scale}")
        if use_ddim:
            tgt_z0 = denoising_utils.run_ddim_p_sample_norm_gs_paper(
                self.ldm,
                src_zT.clone(),
                context,
                NUM_DDIM_STEPS=50,
                num_inference_steps=50,
                start_time=50,
                guidance_scale=guidance_scale,
            )
        else:
            tgt_z0 = denoising_utils.run_pndm_p_sample_norm_gs_paper(
                self.ldm,
                src_zT.clone(),
                context,
                self.PNDM_scheduler,
                NUM_DDIM_STEPS=50,
                num_inference_steps=50,
                start_time=50,
                guidance_scale=guidance_scale,
            )

        # Convert target latent to image
        tgt_x0 = img_utils.latent2im(self.ldm, tgt_z0)
        tgt_x0 = (
            (einops.rearrange(tgt_x0.detach(), "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        # Perform color transfer
        result_image = img_utils.color_transfer(tgt_x0[0], src_x0)

        logger.success("Text modulation completed successfully")
        return result_image


def main():
    # Example usage
    input_image = cv2.imread("001_i_s.png")  # Load your input image here
    src_prompt = "Peter"
    tgt_prompt = "World"

    # Initialize the model
    text_modulator = DBESTTextModulator(
        pretrained_weight_path="ft_et_text_syntext4chars_100k.ckpt",
        yaml_file="train_abinet.yaml",
        checkpoint_tr="best-train-abinet.pth",
        device="cpu",
        seed=888,
    )

    # Run inference
    output_image = text_modulator.infer(
        input_image=input_image,
        src_prompt=src_prompt,
        tgt_prompt=tgt_prompt,
        guidance_scale=0.3,
        use_ddim=True,
        target_size=256,
        finetune_iterations=1500,
        optimization_iterations=1000,
    )
    print("Inference completed successfully")
    cv2.imwrite("output_image.png", output_image)  # Save the output image


if __name__ == "__main__":
    main()
