from diffusers import UNet2DModel
class CustomUNet2DModel(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, timestep, sample, class_labels=None, return_dict=True):
        return super().forward(sample, timestep, class_labels, return_dict).sample