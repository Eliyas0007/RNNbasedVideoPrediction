import cv2
import numpy
import torch
import Models.AutoEncoder as AutoEncoder


from tkinter import *
from einops import rearrange
from PIL import ImageTk, Image



def print_value(value):
    latent = []
    for scale in scales:
        latent.append(scale.get())
    tensor = torch.from_numpy(numpy.array(latent, dtype=numpy.float32)).unsqueeze(0).unsqueeze(0)
    image = ae.decoder(tensor)
    image = rearrange(image.squeeze(0), 'c h w -> h w c')
    image = image.cpu().detach().numpy()
    image = numpy.array(numpy.abs(image) * 256, dtype=numpy.uint8)

    scale_percent = 200 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height) 
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image = cv2.imwrite('./manipulated_image.png', image)
    image = ImageTk.PhotoImage(Image.open("./manipulated_image.png"))
    label.configure(image=image)
    label.image = image


master = Tk()
master.title('Latent Manipulator')
# master.geometry("880x650")
frame = Frame(master, width=64, height=64)
frame.grid(row=8, column=9)
img = ImageTk.PhotoImage(Image.open("simplemovement/example0.png"))
label = Label(frame, image=img)
label.pack()

scales = []

for i in range(16):

    for j in range(8):

        w = Scale(master, from_=0-100.0, to=100.0, orient=HORIZONTAL, command=print_value)
        w.grid(row=i, column=j)
        scales.append(w)

# Model path
ae_model_path = 'workingModels/autoencoder_step18000.pth'

# variational autoencoder
in_channels = 1
latent_size = 128

# simple autoencoder
ae_encoder = AutoEncoder.Encoder(in_channels, latent_size)
ae_decoder = AutoEncoder.Decoder(latent_size)
ae = AutoEncoder.AutoEncoder(ae_encoder, ae_decoder)
ae.load_state_dict(torch.load(ae_model_path, map_location=torch.device('cpu')))

sample_image_path = './simplemovement/example0.png'
sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
sample_image = numpy.asarray(sample_image, dtype=numpy.float32)
sample_tensor = torch.tensor(sample_image).unsqueeze(0).unsqueeze(0)
sample_latent = ae.encoder(sample_tensor).squeeze(0).cpu().detach().numpy()

for i, scale in enumerate(scales):
    scale.set(sample_latent[i])

mainloop()