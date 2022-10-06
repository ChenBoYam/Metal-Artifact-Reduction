import torch
class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 1]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255 / torch.sqrt(mse))

class MAE:
    def __init__(self):
        self.name = "MAE"

    @staticmethod
    def __call__(img1, img2):
        return torch.mean(torch.abs(img1 - img2))