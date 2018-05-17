from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2

pathImage = "/home/geovane/Imagens/teste/imagens/treino"

train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        pathImage,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        save_to_dir="/home/geovane/Imagens/teste/SaidaKeras")

i = 0
for batch in train_generator:

    i += 1
    if i > 19: # save 20 images
        break  # otherwise the generator would loop indefinitely