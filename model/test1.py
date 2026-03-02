import tensorflow as tf

model = tf.keras.models.load_model("revuu.keras", compile=False)
print(model.summary())
