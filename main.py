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

def create_image(imgs):
    img_list = []
    for counter, img in enumerate(imgs):
        curr_img = Image.new('RGB', size=(width, height))
        curr_img.paste(img)
        curr_img.save('Images/'+ prompt.replace(' ', '_') + str(counter) + '.png')
    return img_list

num_images = int(input('Geben Sie die Anzahl an Bildern ein. '))
width = int(input('Geben Sie die Breite der Bilder ein. '))
height = int(input('Geben Sie die Höhe der Bilder ein. '))
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

prompt = input('Geben Sie Ihren Prompt ein ')

with torch.autocast("cuda"):
    images = pipe(
        [prompt] * num_images,
        guidance_scale=7.5,
        latents = latents,
    )['images']
counter = 0
counter = str(counter)
list_images = create_image(images)
# print(images)
# pil_images = image_grid(images, 1, 2)
# pil_images.show()
# pil_images.save(fp=fp)

from deepface import DeepFace
import os

def main(img_filepath1, img_filepath2, dataset_path):
    print("start")
    backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe'
    ]

    # # #face verification
    # obj = DeepFace.verify(img1_path = img_filepath1, enforce_detection=False,
    #                       img2_path = img_filepath2,
    #                       detector_backend = backends[4]
    #                       )
    #
    # # #face recognition
    # df = DeepFace.find(img_path = img_filepath1, enforce_detection=False,
    #                    db_path = dataset_path,
    #                    detector_backend = backends[4]
    #                    )
    #
    #  #embeddings
    # embedding = DeepFace.represent(img_path = img_filepath1,enforce_detection=False,
    #          detector_backend = backends[4]
    #  )
    #
    #facial analysis
    demography = DeepFace.analyze(img_path = img_filepath1, enforce_detection=False,
            detector_backend = backends[4], actions = ['gender']
    )
    #
    # # face detection and alignment
    # face = DeepFace.detectFace(img_path = img_filepath1,enforce_detection=False,
    #        target_size = (224, 224),
    #       detector_backend = backends[4]
    # )
    print(demography)
    #print(demography['gender'])
    output_list = []
    for i in range(len(demography)):
        output_list.append(demography[i]['gender']['Woman'])
        output_list.append(demography[i]['gender']['Man'])
    return output_list



for count, file in enumerate(os.listdir('Images')):
    output = main('Images/'+ file, "", "")
    print(output)
    with open(prompt.replace(' ', '_') + '.txt', 'a') as f:
        if count == 0:
            for i in range(0, len(output), 2):
                f.write('Woman ' + str(output[i]) + ' Man '  +  str(output[i +1]) + ' ')
        else:
            f.write('\n')
            for j in range(0, len(output), 2):
                f.write('Woman ' + str(output[j]) +  ' Man ' + str(output[j + 1]) + ' ')