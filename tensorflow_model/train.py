import tensorflow as tf
import pandas as pd
import math

# 设置数据集的形状
timesteps = 128
lr = 1e-4
num_epochs = 200
batch_size = 4
input_dim = 2
num_classes = 4

# kernel_regularizer=tf.keras.regularizers.l2(1e-2)

# class Model(tf.keras.Model):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.lstm1 = tf.keras.layers.LSTM(16, input_shape=(128, input_dim), return_sequences=True, kernel_regularizer=kernel_regularizer)
#         self.lstm2 = tf.keras.layers.LSTM(32, kernel_regularizer=kernel_regularizer)
#         self.fc1 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=kernel_regularizer)
#         self.fc2 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=kernel_regularizer)

#     def call(self, x):

#         x = self.lstm1(x)
#         x = self.lstm2(x)
#         x = self.fc1(x)
#         x = self.fc2(x)

#         return x

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1,3), strides=(1,1), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1,3), strides=(1,1), padding='same', activation='relu')
        # self.lstm = tf.keras.layers.LSTM(units=16, activation='relu', kernel_regularizer=kernel_regularizer)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(16, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.expand_dims(x, axis=2)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = tf.squeeze(x, axis=2)
        # x = tf.transpose(x, perm=[0, 2, 1])
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == '__main__':

    train_x_pd = pd.read_csv('train_x.csv', header=None)
    train_y_pd = pd.read_csv('train_y.csv', header=None)
    test_x_pd = pd.read_csv('test_x.csv', header=None)
    test_y_pd = pd.read_csv('test_y.csv', header=None)

    train_x = tf.convert_to_tensor(train_x_pd.to_numpy(), dtype=tf.float32)
    train_x = tf.reshape(train_x, [-1, timesteps, input_dim])
    train_y = tf.convert_to_tensor(train_y_pd.to_numpy(), dtype=tf.int32)

    test_x = tf.convert_to_tensor(test_x_pd.to_numpy(), dtype=tf.float32)
    test_x = tf.reshape(test_x, [-1, timesteps, input_dim])
    test_y = tf.convert_to_tensor(test_y_pd.to_numpy(), dtype=tf.int32)

    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_data = train_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_data = test_data.batch(batch_size)

    train_size = tf.data.experimental.cardinality(train_data).numpy()
    test_size = tf.data.experimental.cardinality(test_data).numpy()
    steps_per_epoch = math.ceil(train_size / batch_size)
    validation_steps = math.ceil(test_size / batch_size)

    train_data = train_data.repeat()
    test_data = test_data.repeat()

    # 构建模型
    model = Model()

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

    # 训练模型
    model.fit(train_data, epochs=num_epochs, steps_per_epoch=steps_per_epoch, validation_data=test_data, validation_steps=validation_steps)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,
    #     tf.lite.OpsSet.SELECT_TF_OPS
    #     ]

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)