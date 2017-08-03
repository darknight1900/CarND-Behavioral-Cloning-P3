# Global imports 
import csv
import cv2
import numpy as np
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf 
# Keras related imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.utils  import plot_model
from keras.models import load_model

flags = tf.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

IMG_H  = 160
IMG_W  = 320
IMG_CH = 3

# To show the steering measument distribution 
def plot_training_data_steering_histogram(measurements):
    measurements = np.array(measurements, dtype=np.float32)
    bins = np.arange(-2.0, 2.0, 0.05)
    hist,bin_edges = np.histogram(measurements, bins=bins)
    plt.bar(bin_edges[:-1], hist, width = 1)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.show()


def parse_csv_file_to_lines(csvfile):
    assert(os.path.exists(csvfile))
    firstline = True;
    csv_lines = []
    with open(csvfile, 'r') as csvdata:
        reader = csv.reader(csvdata)
        for line in reader:
            if firstline:
                firstline = False
                continue                  
            csv_lines.append(line)
    return csv_lines

def parse_csv_lines_to_images_steers(csv_lines, img_path):
    center_images = []
    left_images = []
    right_images = []
    measurements = []
    firstline = True;
    for line in csv_lines:
        if firstline:
            firstline = False
            continue        
        # Centre camera image 
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = img_path + filename
        image = cv2.imread(current_path)
        center_images.append(image)
        # Left camera image 
        source_path = line[1]
        filename = source_path.split('/')[-1]
        current_path = img_path + filename
        image = cv2.imread(current_path)
        left_images.append(image)
        # Right camera image
        source_path = line[2]
        filename = source_path.split('/')[-1]
        current_path = img_path + filename
        image = cv2.imread(current_path)
        right_images.append(image)
        
        steer = float(line[3])
        measurements.append(steer)
    return center_images, left_images, right_images, measurements

def augment_randomize_brightness(image, v_lo_ratio, v_hi_ratio):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Convert to numpy array
    hsv = np.array(hsv, dtype=np.float64)
    v_adjust_ratio = np.random.uniform(v_lo_ratio, v_hi_ratio)
    hsv[:,:,2] = hsv[:,:,2] * v_adjust_ratio
    hsv[:,:,2][hsv[:,:,2] > 255] = 255
    # Convert back to integer image 
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Randomly shift image horizontally and vertically 
def augment_randomize_horizontal_vertical_shift(image, measurement, shift_range=100):
    height = image.shape[0]
    width = image.shape[1]
    # Do not allow too large shift 
    shift_range = max(shift_range, width/2)
    # Adjust steering angle 0.0035 per horizontal shit
    shift_x = int(shift_range*np.random.uniform()-shift_range/2)
    measurement = measurement + 0.0035 * shift_x
    # Shift less in vertical direction
    shift_range = int(shift_range/3)
    shift_y = int(shift_range*np.random.uniform()-shift_range/2)
    transform_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    # numpy array store image in the (height, width, channel) order
    # CV2 need (width, height) order
    image = cv2.warpAffine(image,transform_matrix, image.shape[:2][::-1])
    return image, measurement

def augment_randomize_flip_image(image, measurement):
    # randomly flip the images
    if np.random.randint(0, 2) == 0:
        image = cv2.flip(image, 1)
        measurement = measurement * (-1.0)
    return image, measurement


def get_img_steering_data(csv_line_data, img_path):
    line = list(csv_line_data)
    assert(len(line) >= 4)
    img_left_center_right = np.random.randint(3)
    shift_angle = 0.0 
    # Get the steering measurement
    measurement = float(line[3])
    if img_left_center_right == 0: # Center img
        shift_angle = 0.0
        source_path = line[0]
        filename = source_path.split('/')[-1]
        cur_img_path = img_path + filename
        # Make sure the image file exist
        assert(os.path.exists(cur_img_path))
    elif img_left_center_right == 1: # Left img 
        # Adjust steer angle by small value
        shift_angle += 0.25
        source_path = line[1]
        filename = source_path.split('/')[-1]
        cur_img_path = img_path + filename
        # Make sure the image file exist
        assert(os.path.exists(cur_img_path))
    else: #img_left_center_right == 2: ## Right img
        # Adjust steer angle by small value
        shift_angle -= 0.25
        source_path = line[2]
        filename = source_path.split('/')[-1]
        cur_img_path = img_path + filename
        # Make sure the image file exist
        assert(os.path.exists(cur_img_path))
    measurement += shift_angle
    image = cv2.imread(cur_img_path)
    # CV will load image as BGR, convert it back to RGB
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # Randomly adjust brightness 
    image = augment_randomize_brightness(image, 0.4, 1.6)
    # Randomly shift in horizontally and vertical direction
    image, measurement = augment_randomize_horizontal_vertical_shift(image, measurement)
    # Randomly flip the image 
    image, measurement = augment_randomize_flip_image(image, measurement)
    return image, measurement

# Data batch generator 
def generate_training_data_batch(csv_lines, img_path, batch_size = 32, small_steering_th = 0.1):
    assert(os.path.exists(img_path))
    data      = csv_lines
    images    = np.zeros((batch_size, IMG_H, IMG_W, IMG_CH))
    steerings = np.zeros(batch_size)

    while 1:
        for batch_idx in range(batch_size):
            line_data = data[np.random.randint(1, len(data))]
            
            found_good_data = 0
            while found_good_data == 0:
                img, steering = get_img_steering_data(line_data, img_path)
                if abs(steering) >= small_steering_th:
                    found_good_data = 1
                    break
                else:
                    # Randomly drop small steering angle data p
                    random_keep = np.random.uniform(0.0, min(small_steering_th * 3.0, 1.0))
                    if random_keep > small_steering_th:
                        found_good_data = 1
            images[batch_idx]    = img
            steerings[batch_idx] = steering
        yield images, steerings
        

USE_SAVED_MODEL   = 0
USE_RECOVERY_DATA = 1
USE_TACK2         = 1

# Training data from udacity (track1)
CSV_PATH_TRACK1   = 'data/track1/driving_log.csv' 
IMG_PATH_TRACK1   = 'data/track1/IMG/'
csv_lines_track1  = parse_csv_file_to_lines(CSV_PATH_TRACK1)
# Training data by recording recovery from side of the road on track 1
CSV_PATH_TRACK1_RECOVERY   = 'data/track1_recovery/driving_log.csv' 
IMG_PATH_TRACK1_RECOVERY   = 'data/track1_recovery/IMG/'
csv_lines_track1_recovery  = parse_csv_file_to_lines(CSV_PATH_TRACK1_RECOVERY)

# Traning data from track 2
CSV_PATH_TRACK2   = 'data/track2/driving_log.csv' 
IMG_PATH_TRACK2   = 'data/track2/IMG/'
csv_lines_track2  = parse_csv_file_to_lines(CSV_PATH_TRACK2)

if USE_SAVED_MODEL == 0:  
    # Use original udacity track 1 training data 
    csv_lines = csv_lines_track1
    # Split training, validation and test set
    X_train_csv, X_valid_csv = train_test_split(csv_lines, test_size=0.2)

    # Build a netural network 
    model = Sequential()
    # Cropping the top and bottom part of image
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    # Preprocess the data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Conv2D(3, kernel_size = (5, 5), strides=(1, 1), activation='elu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    model.add(Conv2D(24, kernel_size = (5, 5), strides=(1, 1), activation='elu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Minimum batch size is 64
    batch_size = max(FLAGS.batch_size, 64)
    steps_per_epoch_train = 4 * len(X_train_csv) / batch_size;
    steps_per_epoch_valid = 4 * len(X_valid_csv) / batch_size;

    train_generator = generate_training_data_batch(X_train_csv, IMG_PATH_TRACK1, FLAGS.batch_size)
    valid_generator = generate_training_data_batch(X_valid_csv, IMG_PATH_TRACK1, FLAGS.batch_size)
    
    history_object = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch_train, validation_data=valid_generator, validation_steps=steps_per_epoch_valid, epochs=8, verbose=1)

    model.save('model_1.h5')
    plot_model(model, to_file='model.png')
    # plt.plot(history_object.history['loss'])
    # plt.plot(history_object.history['val_loss'])

    # plt.title('model mean squared error loss')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.show()

if USE_RECOVERY_DATA == 1:
    csv_lines = csv_lines_track1_recovery
    # Split training, validation and test set
    X_train_csv, X_valid_csv = train_test_split(csv_lines, test_size=0.2)

    # Split training, validation 
    model = load_model('model_1.h5')

    # Minimum batch size is 64
    batch_size = max(FLAGS.batch_size, 64)

    steps_per_epoch_train  = 8 * len(X_train_csv)/batch_size;
    steps_per_epoch_valid  = 8 * len(X_valid_csv)/batch_size;
    
    track2_train_generator = generate_training_data_batch(X_train_csv, IMG_PATH_TRACK1_RECOVERY, batch_size)
    track2_valid_generator = generate_training_data_batch(X_valid_csv, IMG_PATH_TRACK1_RECOVERY, batch_size)
    
    history_object = model.fit_generator(track2_train_generator, steps_per_epoch=steps_per_epoch_train, validation_data=track2_valid_generator, validation_steps=steps_per_epoch_valid, epochs=8, verbose=1)

    model.save('model_2.h5')

if USE_TACK2 == 1:
    csv_lines = csv_lines_track2
    # Split training, validation and test set
    X_train_csv, X_valid_csv = train_test_split(csv_lines, test_size=0.2)

    # Split training, validation 
    model = load_model('model_2.h5')

    # Minimum batch size is 64
    batch_size = max(FLAGS.batch_size, 64)
    steps_per_epoch_train  = 4*len(X_train_csv);
    steps_per_epoch_valid  = 4*len(X_valid_csv);
    
    track2_train_generator = generate_training_data_batch(X_train_csv, IMG_PATH_TRACK2, batch_size)
    track2_valid_generator = generate_training_data_batch(X_valid_csv, IMG_PATH_TRACK2, batch_size)
    
    history_object = model.fit_generator(track2_train_generator, steps_per_epoch=steps_per_epoch_train, validation_data=track2_valid_generator, validation_steps=steps_per_epoch_valid, epochs=8, verbose=1)

    model.save('model.h5')



