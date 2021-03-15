import tensorflow as tf
import keras
import matplotlib.pyplot as plt

model = keras.models.load_model('./saved_model/20210202/')

img_path_one = "PATH_IMAGE_ONE_TO_TEST"
img_path_two = "PATH_IMAGE_TWO_TO_TEST"

img_one = tf.io.read_file(img_path_one)
img_one = tf.image.convert_image_dtype(tf.io.decode_png(img_one, channels=3), dtype='float32')  # * 1./255
img_one = tf.image.resize(img_one, (224, 224), method=tf.image.ResizeMethod.BILINEAR)
img_one_final = tf.expand_dims(img_one, 0)

img_two = tf.io.read_file(img_path_two)
img_two = tf.image.convert_image_dtype(tf.io.decode_png(img_two, channels=3), dtype='float32')  # * 1./255
img_two = tf.image.resize(img_two, (224, 224), method=tf.image.ResizeMethod.BILINEAR)
img_two_final = tf.expand_dims(img_two, 0)

true_lab = 0

y_pred = tf.round(model.predict([img_one_final, img_two_final])).numpy().flatten()

print("TRUE {} PRED {}".format(true_lab, y_pred))

if (true_lab == y_pred) and true_lab == 0:
    similarity = 'NOT SIMILAR'
elif (true_lab == y_pred) and true_lab == 1:
    similarity = 'SIMILAR'
else:
    similarity = 'NETWORK WRONG MATCH'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

ax1.imshow(img_one[:, :, :].numpy())
ax2.imshow(img_two[:, :, :].numpy())
plt.suptitle("The two tested objects are " + str(similarity), y=0.7)
plt.show()
