import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  rotation_range=40,
                                   fill_mode='nearest',
                                  )
train_generator = train_datagen.flow_from_directory('training',
                                                   target_size=(300,300),
                                                   batch_size=128,
                                                   class_mode='categorical')

                                  
validation_generator = train_datagen.flow_from_directory('validation',
                                                   target_size=(300,300),
                                                   batch_size=128,
                                                   class_mode='categorical')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(60, activation='softmax')
])
model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['acc']
    )

history = model.fit(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)
