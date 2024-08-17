# Image Classification Scripts

This repository contains three different implementations of image classification using deep learning techniques. Each script uses a different framework or approach to classify images based on folder names as labels.

## Scripts Overview

1. `keras_image_classifier.py`: A simple CNN image classifier using Keras/TensorFlow.
2. `pytorch_image_classifier.py`: A simple CNN image classifier using PyTorch.
3. `pytorch_image_classifier_resnet.py`: An advanced CNN image classifier using PyTorch with transfer learning.
4. `keras_image_classifier_resnet.py`: An advanced CNN image classifier using PyTorch with transfer learning.

## Setting Up Your Image Dataset

For the image classification scripts to work properly, you need to organize your image dataset in a specific way. The structure depends on which script you're using:

### For keras_image_classifier.py and pytorch_image_classifier.py

Organize your images into folders, where each folder name represents a class label:

```
data_dir/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

### For pytorch_advanced_image_classifier.py

Split your dataset into training and testing sets:

```
data_dir/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
```

## Usage

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/image-classification-scripts.git
   cd image-classification-scripts
   ```

2. Install the required dependencies (you may want to use a virtual environment):
   ```
   pip install tensorflow torch torchvision numpy
   ```

3. Prepare your dataset as described above.

4. Update the `data_dir` variable in the script you want to use with the path to your dataset.

5. Run the desired script:
   ```
   python keras_image_classifier.py
   # or
   python keras_image_classifier_resnet.py
   # or
   python pytorch_image_classifier.py
   # or
   python pytorch_image_classifier_resnet.py
      
   ```

## Customization

Feel free to modify the scripts to suit your needs. You can adjust parameters such as:

- Image dimensions
- Batch size
- Number of epochs
- Model architecture
- Learning rate and optimizer

## Contributing

Contributions to improve the scripts or add new features are welcome. Please feel free to submit a pull request or open an issue if you have any questions or suggestions.

## License

This project is open-source and available under the MIT License.
