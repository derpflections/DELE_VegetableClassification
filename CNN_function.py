import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow.keras.losses import sparse_categorical_crossentropy
import os
import random
import pathlib
import warnings
import itertools
import math
from sklearn.metrics import accuracy_score
from tensorflow.keras.losses import sparse_categorical_crossentropy

# ignoring redundant warnings
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)


def random_image(dataset, class_names):
    fig, ax = plt.subplots(3, 5, figsize=(20, 15))
    for axs in list(itertools.product([0, 1, 2], [0, 1, 2, 3, 4])):
        for image, label in dataset.take(1):
            ax[axs].imshow(image.numpy())
            ax[axs].set_title(f"Label: {class_names[label.numpy()]}")
            break
    fig.subplots_adjust(bottom=0, left=0, right=1)
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def class_image(train_data, class_names, cmap=None):
    class_images = {name: None for name in class_names}

    for image, label in train_data:
        label_index = label.numpy()
        if label_index.ndim > 0:  # Check if label_index is not a scalar
            label_index = label_index[0]  # Assuming batch size is 1

        class_name = class_names[label_index]
        if class_images[class_name] is None:
            # Convert the image to a numpy array
            class_images[class_name] = image.numpy()

        if all(image is not None for image in class_images.values()):
            break

    # Create a grid for subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 15))

    for (class_name, image), ax in zip(class_images.items(), axes.flatten()):
        if cmap is not None:
            ax.imshow(image, cmap=cmap)
            ax.set_title(class_name)
            ax.axis('off')
        else:
            ax.imshow(image)
            ax.set_title(class_name)
            ax.axis('off')

    plt.suptitle("One image from each class in the dataset.")
    plt.tight_layout()
    plt.show()


def average_images_per_class(train_data, class_names):
    # Dictionary to keep the sum of all images for each class and a count of images
    class_sums = {name: (tf.zeros_like(
        next(iter(train_data))[0]), 0) for name in class_names}

    # Iterate over the dataset and sum up the images tensor-wise for each class
    for image, label in train_data:
        class_name = class_names[label.numpy()]
        sum_images, count = class_sums[class_name]
        class_sums[class_name] = (sum_images + image, count + 1)

    class_averages = {name: (image_sum / count)
                      for name, (image_sum, count) in class_sums.items()}

    # Create a grid for subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 15))
    for ax, (class_name, avg_image) in zip(axes.flatten(), class_averages.items()):
        ax.imshow(avg_image, cmap = "gray")
        ax.set_title(f"Average image for {class_name}")
        ax.axis('off')  # Turn off axis

    plt.suptitle("Average of each class in the dataset")
    plt.tight_layout()
    plt.show()


def average_image(train_data):
    sum_images = None
    count = 0

    for image, _ in train_data:
        if sum_images is None:
            sum_images = tf.zeros_like(image)
        sum_images += image
        count += 1

    average_image = sum_images / count

    plt.figure(figsize=(10, 10))
    plt.imshow(average_image)
    plt.axis('off')  # Turn off the axis
    plt.title("Average Image of the Dataset")
    plt.show()


def image_count_graph(df, ax, set_name):
    sorted_df = df.sort_values('count', ascending=False)
    palette = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
    ax = sns.barplot(x='name', y='count', data=sorted_df,
                     ax=ax, palette=palette(sorted_df['count']))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
    ax.set_xlabel('Vegetable Name')
    ax.set_ylabel('Count')
    ax.set_title(f'Count of different vegetables in {set_name} dataset')

    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                int(sorted_df['count'].iloc[i]), ha='center', va='bottom')

    return ax


def compute_running_mean_std(train_data):
    sum_images = 0.0
    sum_squared_images = 0.0
    n_pixels = 0

    for image, _ in train_data:
        image = tf.cast(image, tf.float32).numpy().flatten()

        sum_images += np.sum(image)
        sum_squared_images += np.sum(np.square(image))

        n_pixels += image.size

    mean_val = sum_images / n_pixels
    std_val = np.sqrt(
        (sum_squared_images - np.square(sum_images) / n_pixels) / n_pixels)

    return mean_val, std_val


def model_metric_graph(history, model, data, class_names, name):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plt.suptitle(f"Metrics for {name} model", y = 1.02)
    fig.tight_layout()

    # creating loss plot
    ax1.plot(history.history['loss'], marker="o")
    ax1.plot(history.history['val_loss'], marker="o")
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss (lower is better)')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Training Dataset', 'Validation Dataset'], loc='upper left')

    # creating accuracy plot
    ax2.plot(history.history['accuracy'], marker="o")
    ax2.plot(history.history['val_accuracy'], marker="o")
    ax2.set_title('Model accuracy')
    ax2.set_ylabel('Accuracy (higher is better)')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Training Dataset', 'Validation Dataset'], loc='upper left')

    fig.subplots_adjust(hspace=0.2)


def model_comparison(model_list, data, model_names):
    accuracy_arr = []
    loss_arr = []
    for model, data_item in zip(model_list, data):
        loss, accuracy = model.evaluate(data_item)
        accuracy_arr.append(accuracy)
        loss_arr.append(loss)

    df = pd.DataFrame({
        'Model Name': model_names,
        'Accuracy': accuracy_arr,
        'Loss': loss_arr
    })


    fig, ax = plt.subplots(2,1, figsize = (12,6), layout = "constrained")
    sns.barplot(x="Accuracy", y="Model Name", data=df, ax=ax[0])
    ax[0].set_title('Model Accuracy Comparison (higher is better)')

    for p in ax[0].patches:
        width = p.get_width()
        ax[0].text(width, p.get_y() + p.get_height() / 2, f'{width*100:.3f}%', 
                   va='center')

    sns.barplot(x="Loss", y="Model Name", data=df, ax=ax[1])
    ax[1].set_title('Model Loss Comparison (lower is better)')

    for p in ax[1].patches:
        width = p.get_width()
        ax[1].text(width, p.get_y() + p.get_height() / 2, f'{width:.3f}', 
                   va='center')

    fig.subplots_adjust(hspace=0.4)