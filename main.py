import torch
from diffusers import StableDiffusionPipeline
print(torch.cuda.is_available())
device = "cuda"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token='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',
).to(device)
#
from PIL import Image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

num_images = 2
width = 400
height =400
#
generator = torch.Generator(device=device)

latents = None
seeds = []
for _ in range(num_images):
    # Get a new random seed, store it and use it as the generator state
    seed = generator.seed()
    seeds.append(seed)
    generator = generator.manual_seed(seed)

    image_latents = torch.randn(
        (1, pipe.unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=device
    )
    latents = image_latents if latents is None else torch.cat((latents, image_latents))

# latents should have shape (4, 4, 64, 64) in this case
print(latents.shape)

prompt = "doctor and nurse"

with torch.autocast("cuda"):
    images = pipe(
        [prompt] * num_images,
        guidance_scale=7.5,
        latents = latents,
    )['images']
fp = 'C:/Users/phili/PycharmProjects/StableDiffusion/Images/images.png'
pil_images = image_grid(images, 1, 2)
pil_images.show()
pil_images.save(fp=fp)