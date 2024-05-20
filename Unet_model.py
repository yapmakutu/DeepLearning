from tensorflow.keras.layers import Layer, Conv2D, Dropout, UpSampling2D, concatenate, Add, Multiply, MaxPool2D, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay

class EncoderBlock(Layer):
    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.pooling = pooling
        self.c1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                         kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))
        self.bn1 = BatchNormalization()
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                         kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))
        self.bn2 = BatchNormalization()
        self.pool = MaxPool2D()

    def call(self, X, **kwargs):
        x = self.c1(X)
        x = self.bn1(x)
        x = self.drop(x)
        x = self.c2(x)
        x = self.bn2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            'rate': self.rate,
            'pooling': self.pooling
        }

class DecoderBlock(Layer):
    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X, **kwargs):
        X, skip_X = X
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            'rate': self.rate,
        }

class AttentionGate(Layer):
    def __init__(self, filters, bn, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.filters = filters
        self.bn = bn
        self.normal = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))
        self.down = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu',
                           kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))
        self.learn = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid', kernel_regularizer=l2(1e-3))
        self.resample = UpSampling2D()
        self.BN = BatchNormalization()

    def call(self, X, **kwargs):
        X, skip_X = X
        x = self.normal(X)
        skip = self.down(skip_X)
        x = Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "bn": self.bn
        }

def build_unet_model(input_shape):
    input_layer = Input(shape=input_shape)

    # Encoder
    encoder1 = EncoderBlock(32, 0.1, name="Encoder1")
    p1, c1 = encoder1(input_layer)

    encoder2 = EncoderBlock(64, 0.1, name="Encoder2")
    p2, c2 = encoder2(p1)

    encoder3 = EncoderBlock(128, 0.2, name="Encoder3")
    p3, c3 = encoder3(p2)

    encoder4 = EncoderBlock(256, 0.2, name="Encoder4")
    p4, c4 = encoder4(p3)

    # Encoding
    encoding_block = EncoderBlock(512, 0.3, pooling=False, name="Encoding")
    encoding = encoding_block(p4)

    # Attention + Decoder
    attention1 = AttentionGate(256, bn=True, name="Attention1")
    a1 = attention1([encoding, c4])

    decoder1 = DecoderBlock(256, 0.2, name="Decoder1")
    d1 = decoder1([encoding, a1])

    attention2 = AttentionGate(128, bn=True, name="Attention2")
    a2 = attention2([d1, c3])

    decoder2 = DecoderBlock(128, 0.2, name="Decoder2")
    d2 = decoder2([d1, a2])

    attention3 = AttentionGate(64, bn=True, name="Attention3")
    a3 = attention3([d2, c2])

    decoder3 = DecoderBlock(64, 0.1, name="Decoder3")
    d3 = decoder3([d2, a3])

    attention4 = AttentionGate(32, bn=True, name="Attention4")
    a4 = attention4([d3, c1])

    decoder4 = DecoderBlock(32, 0.1, name="Decoder4")
    d4 = decoder4([d3, a4])

    # Output
    output_layer = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(d4)

    # Model
    model = Model(
        inputs=[input_layer],
        outputs=[output_layer]
    )

    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', MeanIoU(num_classes=2)]
    )

    return model
