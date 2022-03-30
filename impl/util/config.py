from os import environ

IMG_HEIGHT = 224
IMG_WIDTH = 224

BATCH_SIZE = 128

MONGO_URI = environ['MONGO_URI']

DOWNSTREAM_EPOCHS = 30

PRETEXT_EPOCHS = [
    15,
    30,
    45,
    60
]

EPSILON = 0.01
