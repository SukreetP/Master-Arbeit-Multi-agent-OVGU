import numpy as np
from numpy.core.defchararray import array
import tensorflow as tf
#from tensorflow.keras import callbacks
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
#Name = "agent_2-{}".format((time()))
#tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))



class DQN2(tf.keras.Model):



    def __init__(
        self,
        state_shape: int,
        num_actions: array,
        learning_rate: float
    ):
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.internal_model = self.build_model()

    def build_model(self) -> tf.keras.Model:
        input_state = Input(shape=(6,))
        x = Dense(units= 64)(input_state)
        x = Activation("sigmoid")(x)
        x = Dense(units=32)(x)
        x = Activation("softmax")(x)
        q_value_pred = Dense(units=self.num_actions)(x)
        model = Model(
            inputs=input_state,
            outputs=q_value_pred
        )
        model.compile(
            loss="huber",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        return model

    def call(
        self,
        inputs: np.ndarray
    ) -> np.ndarray:


        #inputs=tf.reshape(inputs, shape=(6,))

        
        return self.internal_model(inputs).numpy()


    def fit(
        self,
        states: np.ndarray,
        q_values: np.ndarray,
    ) -> None:
        self.internal_model.fit(
            x=states,
            y=q_values,
            verbose=1,
        )
        x1 = self.internal_model.fit(x=states, y=q_values, verbose=1)
        loss_history_a1 = x1.history["loss"]
        a1 = np.array(loss_history_a1)
        numpy_loss_history_a1 = list(a1)
        file_a2 = open("loss_history_a2.txt", "a")
        print(*numpy_loss_history_a1, file=file_a2)
        file_a2.close()

    def update_model(
        self,
        other_model: tf.keras.Model
    ) -> None:
        self.internal_model.set_weights(other_model.get_weights())

    def load_model(
        self,
        path: str
    ) -> None:
        self.internal_model.load_weights(path)

    def save_model(
        self,
        path: str
    ) -> None:
        self.internal_model.save_weights(path)


if __name__ == "__main__":
    d = DQN2(
        state_shape=[6],
        num_actions=3,
        learning_rate=0.001
    )
    d.internal_model.summary()
