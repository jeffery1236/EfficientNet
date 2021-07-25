import io
import json
import streamlit as st
import requests
from PIL import Image
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
from torchvision import models, transforms
# import matplotlib.pyplot as plt

def fetch(url: str):
    import requests, os, hashlib, tempfile
    # init file stream
    tempdir = './temp/'
    if not os.path.exists(tempdir):
        os.mkdir(tempdir)
    fp = os.path.join(tempdir, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp) and os.stat(fp).st_size > 0: # check for cached file
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        print(f'fetching {url}')
        dat = requests.get(url).content
        with open(fp+'.tmp', 'wb') as f:
            f.write(dat)
    return dat

def preprocess_img(img) -> torch.tensor:
    aspect_ratio = img.size[0] / img.size[1]
    img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

    arr = np.array(img, dtype=np.float32)
    y0,x0=(np.asarray(arr.shape)[:2]-224)//2
    arr = arr[y0:y0+224, x0:x0+224]
    # plt.imshow(arr)
    # plt.show()

    # Normalize according to https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/simple/example.ipynb
    arr = np.moveaxis(arr, [2,0,1], [0,1,2])
    arr = arr.astype(np.float32)[:3].reshape(1,3,224,224)
    arr /= 255.0
    arr -= np.array([0.485, 0.456, 0.406]).reshape((1,-1,1,1))
    arr /= np.array([0.229, 0.224, 0.225]).reshape((1,-1,1,1))

    return torch.tensor(arr, dtype=torch.float32)


st.markdown("# Whats that thing in the picture? &nbsp")
st.markdown('We will use the EfficientNet Convolutional Neural Network to find out!')
st.text('')

img_url = st.text_input("Enter image url:", 'https://i.natgeofe.com/n/e66e93af-1ad0-4201-8641-ef384108536a/orca.jpg')
if not img_url.startswith('http'):
    st.text('Invalid image url')

img = Image.open(io.BytesIO(fetch(img_url))) # read fetched content into bytes stream and open image
st.image(img)
# img.show()

model_name = 'efficientnet-b0'
model = EfficientNet.from_pretrained(model_name)
img_size = model.get_image_size(model_name)
# model = models.resnet18()

# arr = np.array(img)
transforms = transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
processed_arr = transforms(img).unsqueeze(0)
# processed_arr = preprocess_img(img)

model.eval()
logits = model(processed_arr)
top5 = torch.topk(logits, k=5).indices.squeeze(0).tolist() # get top5 label ids

with open("./imagenet1000_labels.txt", 'r') as f:
    labels_map = json.load(f)

for idx in top5:
    label = labels_map[str(idx)]
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    # print('{:<75} ({:.2f}%'.format(label, prob*100))
    st.text('{:<75} ({:.2f}%)'.format(label, prob*100))

