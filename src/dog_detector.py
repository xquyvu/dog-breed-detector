import cv2
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms


class DogDetector:
    def __init__(self):
        # define VGG16 model
        VGG16 = models.vgg16(pretrained=True)

        # move model to GPU
        self.VGG16 = VGG16.cuda()

    def VGG16_predict(self, img_path):
        '''
        Use pre-trained VGG-16 model to obtain index corresponding to
        predicted ImageNet class for image at specified path

        Args:
            img_path: path to an image

        Returns:
            Index corresponding to VGG-16 model's prediction
        '''

        img = Image.open(img_path)

        transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        img_tensor = transformer(img).unsqueeze_(0).cuda()

        model_output = self.VGG16(img_tensor)

        prediction = torch.argmax(model_output).item()

        return prediction

    def detect_dog(self, img_path):
        return True if 151 <= self.VGG16_predict(img_path) <= 268 else False
