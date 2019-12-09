import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt
from PIL import ImageFile, Image
from src.dog_detector import DogDetector
from src.face_detector import FaceDetector
import sys


def load_model(state_dict='model_transfer.pt'):
    # Load model - VGG16
    model_transfer = models.vgg16(pretrained=False)

    # Replace the last layer
    n_inputs = model_transfer.classifier[6].in_features
    model_transfer.classifier[6] = nn.Linear(n_inputs, 133)

    # Use gpu
    model_transfer = model_transfer.cuda()

    # Load weights
    model_transfer.load_state_dict(torch.load(state_dict))

    return model_transfer


def predict_breed_transfer(model_transfer, class_names, img_path):
    # load the image and return the predicted breed
    img = Image.open(img_path)

    transformer = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    img_tensor = transformer(img).unsqueeze_(0).cuda()

    model_output = model_transfer(img_tensor)

    prediction = torch.argmax(model_output).item()

    return class_names[prediction]


def get_name_from_path(img_path):
    return img_path.split('/')[-1].split('.')[0].rsplit('_', 1)[0]


def main(img_path, state_dict_path='model_transfer.pt'):
    # Initialise
    dog_detector = DogDetector()
    face_detector = FaceDetector()

    model_transfer = load_model(state_dict=state_dict_path)

    with open('./class_names.txt') as f:
        class_names = f.read().split('\n')

    # Show image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

    if dog_detector.detect_dog(img_path):
        plt.title(''.join([
            'The algorithms think the dog breed is ', predict_breed_transfer(model_transfer, class_names, img_path),
            '\n',
            'The correct dog breed is ', get_name_from_path(img_path)
        ]))

    elif face_detector.detect_face(img_path):
        plt.title('The algorithm thinks ' + get_name_from_path(img_path) + ' looks like a ' + predict_breed_transfer(model_transfer, class_names, img_path))

    else:
        plt.title('The algorith detects neither dog nor human')

    plt.show()


if __name__ == '__main__':
    main(sys.argv[0])
