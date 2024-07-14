import tensorflow as tf
import pandas as pd

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1,3), strides=(1,1), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1,3), strides=(1,1), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,3), strides=(1,1), padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(32, activation='relu')
        self.fc2 = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs):
        x = tf.transpose(inputs, perm=[0, 2, 1])  # Convert (batch, seq, channels) to (batch, channels, seq)
        x = tf.expand_dims(x, axis=2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
    train_x = tf.reshape(train_x, [-1, 64, 6])
    train_y = tf.convert_to_tensor(train_y_pd.to_numpy(), dtype=tf.int32)

    test_x = tf.convert_to_tensor(test_x_pd.to_numpy(), dtype=tf.float32)
    test_x = tf.reshape(test_x, [-1, 64, 6])
    test_y = tf.convert_to_tensor(test_y_pd.to_numpy(), dtype=tf.int32)

    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_data = train_data.batch(5)

    test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_data = test_data.batch(5)

    train_data = train_data.repeat()
    test_data = test_data.repeat()

    # 设置数据集的形状
    timesteps = 64
    input_dim = 6
    num_classes = 3
    lr = 1e-5

    # 构建模型
    model = Model()

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

    # 训练模型
    model.fit(train_data, epochs=1000, steps_per_epoch=6, validation_data=test_data, validation_steps=3)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)