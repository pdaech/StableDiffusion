import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from deepface import DeepFace
import os




def create_image(imgs, prompt, counter, width, height):
    curr_img = Image.new('RGB', size=(width, height))
    curr_img.paste(imgs[0])
    curr_img.save('Images/'+ prompt.replace(' ', '_') + str(counter) + '.png')

def use_pipeline(prompt):
    generator = torch.Generator(device=device)
    latents = None
    seeds = []
    seed = generator.seed()
    seeds.append(seed)
    generator = generator.manual_seed(seed)

    image_latents = torch.randn(
        (1, pipe.unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=device
    )
    latents = image_latents if latents is None else torch.cat((latents, image_latents))
    with torch.autocast("cuda"):
        images = pipe(
            [prompt] * 1,
            guidance_scale=7.5,
            latents = latents,
        )['images']
    return images


# print(images)
# pil_images = image_grid(images, 1, 2)
# pil_images.show()
# pil_images.save(fp=fp)


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

def analyze(prompt):
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
print(torch.cuda.is_available())
device = "cuda"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token='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',
).to(device)

num_images = int(input('Anzahl an Bildern. '))
width = int(input('Geben Sie die Breite der Bilder ein. '))
height = int(input('Geben Sie die Höhe der Bilder ein. '))

prompt = input('Geben Sie Ihren Prompt ein ')

for i in range(num_images):
    curr_imgs = use_pipeline(prompt)
    create_image(curr_imgs, prompt, i, width, height)


analyze(prompt)

