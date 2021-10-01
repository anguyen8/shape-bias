from PIL import Image
import os
from tqdm import tqdm

def resize_img(dir, shape=(224,224)):
    imageList = os.listdir(dir)
    for image_file in imageList:
        f_dir = os.path.join(dir, image_file)
        with Image.open(f_dir) as image:
            img = image.resize(shape)
            img.save(f_dir)

if __name__ == "__main__":
    paths = os.listdir("imagenet_texture_1bb")
    # for path in paths:
    #     print("Resizing "+path.split("/")[-1])
    #     resize_img(path)

    tqdm_path = tqdm(paths)
    for path in tqdm_path:
        dir = os.path.join("imagenet_texture_1bb", path)
        resize_img(dir)
