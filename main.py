import tensorflow as tf
import tensorflow_datasets as tfds

# Load the EMNIST "balanced" dataset, which includes digits and letters
(ds_train, ds_test), ds_info = tfds.load('emnist/balanced', split=['train', 'test'], as_supervised=True, with_info=True)

# Preprocess the data (normalize and resize)
def preprocess(image, label):
    image = tf.image.resize(image, [28, 28])  # Resize to 28x28
    image = image / 255.0                      # Normalize to [0,1] range
    return image, label

ds_train = ds_train.map(preprocess).batch(128).shuffle(10000)
ds_test = ds_test.map(preprocess).batch(128)

from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(47, activation='softmax')  # 47 classes for digits and letters
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(ds_train, epochs=5, validation_data=ds_test)

#Saving model
model.save('emnist_model.h5')



# model_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('emnist_model.h5')

# Initialize FastAPI app
app = FastAPI()

class ImageData(BaseModel):
    image: list  # Expecting a 28x28 flattened image as a list

@app.post("/predict")
async def predict(data: ImageData):
    try:
        # Convert list to numpy array and reshape
        image_array = np.array(data.image).reshape(1, 28, 28, 1)
        # Normalize image data
        image_array = image_array / 255.0
        # Predict using the model
        predictions = model.predict(image_array)
        # Get the class with the highest probability
        predicted_class = int(np.argmax(predictions[0]))
        return {"predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)