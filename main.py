#DENSENET169 Model used
import time

from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

sttime=time.time()#store start time

#setting physical device
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)


train_df = pd.read_csv("g4g1.csv")
test_df = pd.read_csv("g4g1test.csv")

IMG_SIZE=1024


def crop_center_square(frame):
				y, x = frame.shape[0:2]
				min_dim = min(y, x)
				start_x = (x // 2) - (min_dim // 2)
				start_y = (y // 2) - (min_dim // 2)
				return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
				cap = cv2.VideoCapture(path)
				frames = []
				try:
								while True:
												ret, frame = cap.read()
												if not ret:
																break
												frame = crop_center_square(frame)
												frame = cv2.resize(frame, resize)
												frame = frame[:, :, [2, 1, 0]]
												frames.append(frame)

												if len(frames) == max_frames:
																break
				finally:
								cap.release()
				return np.array(frames)


def extract_feat():
				feature_extractor = keras.applications.densenet.DenseNet169(
								weights="imagenet",
								include_top=False,
								pooling="avg",
								input_shape=(IMG_SIZE, IMG_SIZE, 3), classifier_activation=None
				)
				preprocess_input=keras.applications.densenet.preprocess_input
				input=keras.Input((IMG_SIZE,IMG_SIZE , 3))
				preprocessed = preprocess_input(input)
				return keras.Model(input, feature_extractor(preprocessed), name="feature_extractor")


feature_extractor = extract_feat()
#converting to labels
label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))
labels = label_processor(train_df["tag"].values[..., None]).numpy()


BATCH_SIZE = 64
EPOCHS = 100

MAX_SEQ_LENGTH = 30
NUM_FEATURES = 2048


def prepare_all_videos(df, root_dir):
				num_samples = len(df)
				video_paths = df["video_name"].values.tolist()
				labels = label_processor(df["tag"].values[..., None]).numpy()
				frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")  # 145,20
				frame_features = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")  # 145,20,2048

				# For each video.


				for idx, path in enumerate(video_paths):

								frames = load_video(os.path.join(root_dir, path))
								frames = frames[None, ...]

								# Initialize placeholders to store the masks and features of the current video.
								temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
								temp_frame_features = np.zeros(
												shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
								)

								# Extract features from the frames of the current video.
								for i, batch in enumerate(frames):
												video_length = batch.shape[0]
												for j in range(min(MAX_SEQ_LENGTH, video_length)):
																temp_frame_features[i, j, :] = feature_extractor.predict(
																				batch[None, j, :]
																)
												temp_frame_mask[i, :min(MAX_SEQ_LENGTH, video_length)] = 1  # 1 = not masked, 0 = masked

								frame_features[idx,] = temp_frame_features.squeeze()
								frame_masks[idx,] = temp_frame_mask.squeeze()

				return (frame_features, frame_masks), labels



#

train_data, train_labels = prepare_all_videos(train_df, "train")
test_data, test_labels = prepare_all_videos(test_df, "test")




# Utility for our sequence model.
def get_sequence_model():
				class_vocab = label_processor.get_vocabulary()

				frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
				mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")


				x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
				x = keras.layers.GRU(8)(x)
				x = keras.layers.Dropout(0.4)(x)
				x = keras.layers.Dense(8, activation="relu")(x)
				output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

				rnn_model = keras.Model([frame_features_input, mask_input], output)

				rnn_model.compile(
								loss="sparse_categorical_crossentropy", optimizer="adamax", metrics=["accuracy"]
				)


				return rnn_model
				#return model




# Utility for running experiments.
def run_experiment():
				checkpoint = keras.callbacks.ModelCheckpoint(
								"./sanskriti/vids", save_weights_only=False, save_best_only=True, verbose=1,save_freq="epoch"
				)

				seq_model = get_sequence_model()
				history = seq_model.fit(
								[train_data[0], train_data[1]],
								train_labels,
								validation_split=0.3,
								epochs=50,
								callbacks=[checkpoint],
				)

				seq_model.load_weights("./sanskriti/vids")
				_, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
				print(f"Test accuracy: {round(accuracy * 100, 2)}%")

				return history, seq_model
from tqdm import tqdm
for i in tqdm(range(1)):


					_, sequence_model = run_experiment()
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    for i in np.argsort(probabilities)[::-1]:
								print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    #none
    return frames

et=time.time()
print(et-sttime)
test_video = np.random.choice(test_df["video_name"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)




