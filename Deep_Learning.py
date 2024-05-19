import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from Unet_model import EncoderBlock, DecoderBlock, AttentionGate

class DeepLearning:
    def __init__(self, unet_model_path, cnn_model_path):
        self.segmentation_model = load_model(unet_model_path,
                                             custom_objects={
                                                 'EncoderBlock': EncoderBlock,
                                                 'DecoderBlock': DecoderBlock,
                                                 'AttentionGate': AttentionGate
                                             })
        self.classification_model = load_model(cnn_model_path)

    def segment_and_classify(self, image_path):
        test_image = cv2.imread(image_path)
        resized_image = cv2.resize(test_image, (256, 256))
        resized_image = np.expand_dims(resized_image, axis=0)

        predicted_mask = self.segmentation_model.predict(resized_image)
        predicted_mask_resized = cv2.resize(predicted_mask[0], (test_image.shape[1], test_image.shape[0]))

        _, binary_mask = cv2.threshold(predicted_mask_resized, 0.5, 1, cv2.THRESH_BINARY)
        binary_mask = (binary_mask * 255).astype(np.uint8)

        predicted_mask_resized_for_cnn = cv2.resize(binary_mask, (256, 256))
        predicted_mask_final = np.expand_dims(predicted_mask_resized_for_cnn, axis=0)
        predicted_mask_final = np.expand_dims(predicted_mask_final, axis=-1)

        prediction = self.classification_model.predict(predicted_mask_final)

        # Save the visual comparison
        self.save_visual_comparison(test_image, binary_mask, image_path)

        return predicted_mask_resized, prediction

    def save_visual_comparison(self, original_image, predicted_mask, original_path):
        mask_overlay = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)  # Convert mask back to BGR for overlay
        mask_overlay = np.where(mask_overlay > 0, (0, 0, 255), original_image)  # Overlay mask in red

        combined_image = np.concatenate((original_image, mask_overlay), axis=1)  # Combine images side-by-side
        save_path = original_path.replace('.png', '_comparison.png')  # Modify the original path for saving

        cv2.imwrite(save_path, combined_image)  # Save the combined image

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    unet_model_path = os.path.join(PROJECT_ROOT, "final_model.h5")  # Ensure this file exists
    cnn_model_path = os.path.join(PROJECT_ROOT, "trained_model_size256.h5")
    image_path = os.path.join(PROJECT_ROOT, "Dataset", "Dataset_BUSI_with_GT_split", "benign", "images", "benign (1).png")

    # Perform segmentation and classification
    deep_learning = DeepLearning(unet_model_path, cnn_model_path)
    predicted_mask, prediction = deep_learning.segment_and_classify(image_path)

    class_names = ["Benign", "Malignant", "Normal"]
    predicted_class = class_names[np.argmax(prediction[0])]

    print("Prediction: ", predicted_class)
