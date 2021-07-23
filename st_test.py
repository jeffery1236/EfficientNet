import io
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

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

st.markdown("# Whats that thing in the picture? &nbsp ")

img_url = st.text_input("Enter image url:", 'https://images.newscientist.com/wp-content/uploads/2020/12/11153312/orca-killer-whale-a2c8ec_web.jpg')
if not img_url.startswith('http'):
    st.text('Invalid image url')

img = Image.open(io.BytesIO(fetch(img_url))) # read fetched content into bytes stream and open image
print(type(img))
# plt.imshow(img)
# plt.show()

st.image(img)
