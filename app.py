
import streamlit as st

import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import regex as re
import tifffile as tiff
from PIL import Image
import seaborn as sns
import albumentations as A
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation = 1):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation,stride=1, bias=False,
                     dilation = dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = dilation, stride = 1, bias=False,
                     dilation = dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features = [64, 128, 256, 512], rates = (1,1,1,1)):
        super(UNet, self).__init__()
        self.down_part = nn.ModuleList()
        self.up_part = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Part
        for i,feature in enumerate(features):
            self.down_part.append(DoubleConv(in_channels, feature, dilation = rates[i]))
            in_channels = feature
        # Decoder Part
        for i,feature in enumerate(reversed(features)):
            self.up_part.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.up_part.append(DoubleConv(2*feature, feature, dilation = rates[i]))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        skip_connections = []
        for down in self.down_part:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.up_part), 2):
            x = self.up_part[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x), dim = 1)
            x = self.up_part[idx + 1](concat_skip)

        return self.output(x)

# load classifier model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels = 3, out_channels = 5).to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/inputs/checkpoint.pth'))
model.eval()


# prediction function


def main():
    st.title("Land Cover AI Model")

    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Land Cover AI App</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Upload image through Streamlit
    uploaded_file_input = st.file_uploader("Choose a input JPG file", type=["jpg", "jpeg"])

    if uploaded_file_input is not None:
        # Display the uploaded image
        st.image(uploaded_file_input, caption="Uploaded Image.", use_column_width=True)

    # Upload image through Streamlit
    uploaded_file_mask = st.file_uploader("Choose a input mask PNG file", type=["png"])

    if uploaded_file_mask is not None:
        # Display the uploaded image
        st.image(uploaded_file_mask, caption="Uploaded Image.", use_column_width=True)

    if st.button("Predict"):
      img = np.transpose(cv2.imdecode(np.fromstring(uploaded_file_input.read(), np.uint8), cv2.IMREAD_COLOR),(2,0,1))
      mask = cv2.imdecode(np.fromstring(uploaded_file_mask.read(), np.uint8), cv2.IMREAD_COLOR)

      label = mask[:,:,1]

      input_tensor = torch.tensor(img, dtype = torch.float32)/255
      lbl_tensor = torch.tensor(label, dtype = torch.int64)


      class_probs = {0: 0, 1: 0, 2: 0, 3: 0, 4:0}
      num_samples = {0:0, 1:0, 2:0, 3:0, 4:0}

      X = input_tensor.to(device)
      y = lbl_tensor.to(device)

      model.eval()

      X_dash = X[None,:,:,:].to(device)

      preds = torch.argmax(model(X_dash), dim = 1)

      preds = torch.squeeze(preds).detach().cpu().numpy()

      st.write("Predicted Mask")

      st.image(preds)

      if not os.path.exists("/content/test"):
        # If not, create the directory
        os.makedirs("/content/test")

      with open(os.path.join("/content/test",uploaded_file_input.name),"wb") as f:
        f.write(uploaded_file_input.getbuffer())
      with open(os.path.join("/content/test",uploaded_file_mask.name),"wb") as f:
        f.write(uploaded_file_mask.getbuffer())

      st.success('The file is saved!')


if __name__=='__main__':
    main()

