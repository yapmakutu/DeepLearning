import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tensorflow as tf

# Test file name is Unet.py
from UNet import EncoderBlock, DecoderBlock, DeepLearning


class TestEncoderBlock(unittest.TestCase):
    def setUp(self):
        self.encoder_block = EncoderBlock(filters=32, rate=0.1, pooling=True)

    def test_call_with_pooling(self):
        # Prepare a dummy input tensor
        dummy_input = tf.random.normal([1, 256, 256, 3])
        # Call the encoder block directly without mocking TensorFlow internals
        output, _ = self.encoder_block(dummy_input)
        # Verify the output shape
        self.assertEqual(output.shape, [1, 128, 128, 32])  # Example expected shape, adjust as necessary


class TestDecoderBlock(unittest.TestCase):
    def setUp(self):
        self.decoder_block = DecoderBlock(filters=32, rate=0.1)

    def test_call(self):
        # Prepare dummy input tensors that mimic expected operational shapes post-upsampling and from the encoder
        # Assuming the upsampling doubles the size, we prepare the skip connection to match the expected upsampled size
        dummy_input = tf.random.normal([1, 64, 64, 32])  # Size before upsampling
        # Skip connection size matches expected upsampled size
        dummy_skip_connection = tf.random.normal([1, 128, 128, 32])

        # Call the decoder block directly
        output = self.decoder_block([dummy_input, dummy_skip_connection])

        # Verify the output shape
        # Assuming the output size matches the skip connection size,
        # and the number of filters in the final Conv2D layer of EncoderBlock
        self.assertEqual(output.shape, [1, 128, 128, 32])  # Adjust as necessary based on your network architecture


class TestDeepLearning(unittest.TestCase):
    @patch('UNet.load_model', autospec=True)
    def setUp(self, mock_load_model):
        mock_load_model.return_value = MagicMock()
        self.deep_learning = DeepLearning('dummy_model_path.h5')

    @patch('UNet.cv2.imread', autospec=True)
    @patch('UNet.cv2.resize', autospec=True)
    @patch('UNet.cv2.threshold', autospec=True)
    @patch('UNet.cv2.imwrite', autospec=True)
    def test_segment(self, mock_imwrite, mock_threshold, mock_resize, mock_imread):
        # Adjust the mocks to ensure array shapes are consistent
        mock_imread.return_value = np.random.rand(512, 512, 3).astype(np.uint8)
        mock_resize.side_effect = lambda image, size: np.random.rand(size[1], size[0], 3).astype(
            np.uint8)  # Fix size ordering
        mock_threshold.return_value = (
            None, np.random.rand(512, 512).astype(np.uint8))  # Adjusted to match original image size for testing

        # Perform the segmentation
        binary_mask = self.deep_learning.segment('dummy_image_path.png')

        # Assertions
        mock_imread.assert_called_once_with('dummy_image_path.png')
        self.assertEqual(mock_resize.call_count, 2)  # Called twice as expected
        mock_threshold.assert_called_once()
        mock_imwrite.assert_called_once()

        # Additional Assertions
        # Verify the binary_mask is a numpy array with expected shape
        self.assertIsInstance(binary_mask, np.ndarray)
        self.assertEqual(binary_mask.shape, (512, 512))  # Expected shape after resizing back


if __name__ == '__main__':
    unittest.main()
