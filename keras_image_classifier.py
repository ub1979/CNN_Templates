# ==============================================================
# Importing the libraries
# ==============================================================
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==============================================================
# Function to create a simple CNN model
# ==============================================================
def create_cnn_model(input_shape, num_classes):
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # Second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Third convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Flatten layer
    model.add(Flatten())

    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


# ==============================================================
# Main function to run the image classifier
# ==============================================================
def main():
    # Set the path to your dataset
    data_dir = 'path/to/your/dataset'

    # Set image dimensions and batch size
    img_height, img_width = 150, 150
    batch_size = 32

    # Create ImageDataGenerator for data augmentation and normalization
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Create train generator
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Create validation generator
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Get the number of classes
    num_classes = len(train_generator.class_indices)

    # Create the CNN model
    model = create_cnn_model((img_height, img_width, 3), num_classes)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    epochs = 10
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    # Print the class labels
    print("Class labels:", train_generator.class_indices)

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Validation accuracy: {val_accuracy:.2f}")


# ==============================================================
# Run the main function
# ==============================================================
if __name__ == "__main__":
    main()