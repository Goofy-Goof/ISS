import os
from numpy import linspace

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 128

MONGO_URI = os.environ['MONGO_URI']

DOWNSTREAM_EPOCHS = [
    10,
    20,
    30,
    40,
    50,
    100
]

OPTIMAL_DOWNSTREAM_EPOCHS = 50
PRETEXT_EPOCHS = [
    25,
    50,
    75,
    100
]

EPSILONS = linspace(start=0.01, stop=0.1, num=19)
