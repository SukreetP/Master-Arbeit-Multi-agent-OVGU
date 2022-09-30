import numpy as np
from numpy.core.defchararray import array
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



class DQN1(tf.keras.Model):



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
        input_state = Input(shape=self.state_shape)
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
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model

    def call(
        self,
        inputs: np.ndarray,
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
            verbose=1
        )

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
    d = DQN1(
        state_shape=[6],
        num_actions=3,
        learning_rate=0.001
    )
    d.internal_model.summary()
