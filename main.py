import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from deepface import DeepFace
import os
import pandas as pd
import re




def create_image(imgs, prompt, counter, width, height):
    curr_img = Image.new('RGB', size=(width, height))
    curr_img.paste(imgs[0])
    curr_img.save('Images/'+ prompt.replace(' ', '_') + '_' + str(counter) + '.png')

def use_pipeline(prompt, input_seed):
    generator = torch.Generator(device=device)
    latents = None
    seeds = []
    if not input_seed:
        seed = generator.seed()
    else:
        seed = input_seed

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
            )
    nsfw = images['nsfw_content_detected']
    images = images['images']
    return images, nsfw


def seed_req(num_images):
    random_seed = input('Random seed? (y/n)')
    if random_seed != 'y':
        seeds = []
        seed_asc = input('Seed ascending? (y / n)')
        if seed_asc != 'y':
            while num_images > 0:
                print('you need' + num_images + 'more seeds')
                seeds.append = input('Which seeds do you want to use? Enter one seed')
                num_images -= 1
        else:
            seeds = range(0, num_images)
        return seeds


# print(images)
# pil_images = image_grid(images, 1, 2)
# pil_images.show()
# pil_images.save(fp=fp)


def main(img_filepath1, img_filepath2, dataset_path):
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
    #print(demography['gender'])
    output_list = []
    for i in range(len(demography)):
        output_list.append(demography[i]['gender']['Woman'])
        output_list.append(demography[i]['gender']['Man'])
    return output_list


def natural_sort_key(s):
    """Function to use as a key for sorting"""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]
def analyze(prompt):
    df = pd.DataFrame()
    all_outputs = []
    lengths = []
    for count, file in enumerate(sorted(os.listdir('Images'), key=natural_sort_key)):
        if file.split('_')[:-1] == prompt.split(' '):
            output = main('Images/'+ file, "", "")
            print(file)
            lengths.append(int(len(output)/2))
            all_outputs.append(output)
    max_length = max(lengths)
    columns = []
    for i in range(0, max_length):
        columns.append(str(i) + ' Woman')
        columns.append(str(i) + ' Man')
    columns = ['num_faces'] + columns
    for idx, elem in enumerate(all_outputs):
        if lengths[idx] < max_length:
            for j in range(int(max_length - lengths[idx])*2):
                elem.append(0.0)
        all_outputs[idx] = [lengths[idx]] + elem
    # print(all_outputs)
    # print(columns)
    df = pd.DataFrame(all_outputs, columns = columns)
    df.to_csv(prompt.replace(' ', '_') + '.csv', index=False)




if input('a to only analyze ') == 'a':
    analyze(input('Filename '))
else:
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
    # seeds = seed_req(num_images)
    # max_images = num_images
    # no_nsfw = 1
    cn = 0
    while num_images > 0:
        print(num_images)
        curr_imgs, nsfw = use_pipeline(prompt, [])
        if nsfw[0]:
            # seeds[num_images] = max_images + no_nsfw
            pass
        elif not nsfw[0]:
            create_image(curr_imgs, prompt, cn, width, height)
            cn += 1
            num_images -=1
    analyze(prompt)

# df2 = pd.read_csv('man.csv') # read csv file
# print(df2)


