# ==============================================================
# Importing the libraries
# ==============================================================
import os
import numpy as np
from tensorflow.keras.applications import ResNet34
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==============================================================
# Function to create a model based on ResNet34
# ==============================================================
def create_resnet34_model(input_shape, num_classes):
    # Load the ResNet34 model, excluding the top layers
    base_model = ResNet34(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze all layers except the last two
    for layer in base_model.layers[:-2]:
        layer.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Combine the base model and custom layers
    model = Model(inputs=base_model.input, outputs=outputs)

    return model


# ==============================================================
# Main function to run the image classifier
# ==============================================================
def main():
    # Set the path to your dataset
    data_dir = 'path/to/your/dataset'

    # Set image dimensions and batch size
    img_height, img_width = 224, 224  # ResNet34 default input size
    batch_size = 32

    # Create ImageDataGenerator for data augmentation and normalization
    datagen = ImageDataGenerator(
        preprocessing_function=ResNet34.preprocess_input,
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

    # Create the ResNet34 model
    model = create_resnet34_model((img_height, img_width, 3), num_classes)

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