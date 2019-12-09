## Overview

In this project, I have developed an application that, given an image of a dog, it will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

## Installation

1. Clone the repository and navigate to the downloaded folder.

```	
git clone https://github.com/xquyvu/dog-breed-detector.git
cd dog-breed-detector
```

2. Get the trained models
- The model can be downloaded [here](https://www.dropbox.com/s/tuctyg6dmmvpy8y/model_transfer.pt?dl=0)
- Save the trained model as `./model_transfer.pt`


## Usage
Simply run main.py and provide the path to your image.

Example:
`python main.py --img_path ./images/Curly-coated_retriever_03896.jpg`
