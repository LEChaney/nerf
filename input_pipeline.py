import tensorflow as tf
import os
import matplotlib.pyplot as plt
import glob
import time
from run_nerf_helpers import get_rays_from_xy_and_poses

from load_llff import load_llff_data

def get_dataset_for_task(task_path, batch_size=1024, max_images=10000, holdout_every=8):
    def random_sample_1d(*x):
        # Randomly sample list of image paths with replacement
        N = tf.shape(x[0])[0]
        i_batch = tf.random.uniform([batch_size], minval=0, maxval=N, dtype=tf.int32)
        # Potentially speed up later processing by grouping identical images next to each other.
        # Note that because the task paths are already shuffled, sorting the indices still leaves
        # images in a random order, but wth copies of the same image grouped next to each other.
        i_batch = tf.sort(i_batch)
        x = tuple(tf.gather(elem, i_batch) for elem in x)
        # tf.print(x)
        return x

    def load_image(image_paths, *passthrough):
        unique_image_paths, indices = tf.unique(image_paths, out_idx=tf.int32)
        def _load_image(im_path):
            im_file = tf.io.read_file(im_path)
            return tf.image.decode_image(im_file) / 255
        unique_images = tf.map_fn(_load_image, unique_image_paths, dtype=tf.float32)
        # tf.print(tf.shape(unique_images))

        return (tf.gather(unique_images, indices),) + passthrough

    def load_and_sample_rgb_and_rays(image_paths, poses, bounds): # Assume image_paths is first element in incoming data
        N = tf.shape(image_paths)[0]

        # Load images (only load unique images)
        unique_image_paths, indices = tf.unique(image_paths, out_idx=tf.int32)
        def load_image(im_path):
            im_file = tf.io.read_file(im_path)
            return tf.image.decode_image(im_file) / 255
        unique_images = tf.map_fn(load_image, unique_image_paths, dtype=tf.float32)
        # tf.print(tf.shape(unique_images))

        # Randomly sample image pixels
        height = tf.shape(unique_images, out_type=tf.int32)[1]
        width = tf.shape(unique_images, out_type=tf.int32)[2]
        y = tf.random.uniform([N], minval=0, maxval=height, dtype=tf.int32)
        x = tf.random.uniform([N], minval=0, maxval=width, dtype=tf.int32)
        target_rgb = tf.gather_nd(unique_images, tf.stack([indices, y, x], axis=-1))

        # Get rays
        focal = poses[:, 2, -1]
        rays_o, rays_d = get_rays_from_xy_and_poses(x, y, height, width, focal, poses)
        rays = tf.stack([rays_o, rays_d], axis=1)

        return (rays, target_rgb, bounds, tf.repeat(height, N), tf.repeat(width, N), focal)

    def load_data_wrapper(task_path):
        task_path = task_path.decode("utf-8")
        return load_llff_data(task_path, factor=None, return_paths_instead_of_img_data=True, spherify=True) #TODO: remove hard coding of arguments
    
    task_path = tf.convert_to_tensor(task_path)
    img_paths, poses, bds, render_poses, i_test = tf.numpy_function(load_data_wrapper, [task_path], (tf.string, tf.float32, tf.float32, tf.float32, tf.int32))

    if not isinstance(i_test, list):
        i_test = [i_test]

    if holdout_every > 0:
        print('Auto holdout every', holdout_every)
        i_test = tf.range(tf.shape(img_paths)[0])[::holdout_every]

    i_val = i_test
    i_train = tf.convert_to_tensor([i for i in tf.range(tf.shape(img_paths)[0]) if
                                   (i not in i_test and i not in i_val)])

    # TESTS #
    # test_img_paths = tf.gather(img_paths, i_train[:10])
    # test_poses = tf.gather(poses, i_train[:10])
    # test_bds = tf.gather(bds, i_train[:10])
    # load_and_sample_rgb_and_rays(test_img_paths, test_poses, test_bds)

    poses_ds = tf.data.Dataset.from_tensor_slices(tf.gather(poses, i_train))
    bds_ds = tf.data.Dataset.from_tensor_slices(tf.gather(bds, i_train))
    paths_ds = tf.data.Dataset.from_tensor_slices(tf.gather(img_paths, i_train))

    train_ds = (tf.data.Dataset.zip((paths_ds, poses_ds, bds_ds))
          .batch(max_images)
          .repeat()
          .map(random_sample_1d)
          .unbatch()
          .batch(512)
          .map(load_and_sample_rgb_and_rays)
          .unbatch()
          .batch(batch_size)
          .prefetch(tf.data.experimental.AUTOTUNE)
    ) 

    test_poses_ds = tf.data.Dataset.from_tensor_slices(tf.gather(poses, i_test))
    test_render_poses = tf.data.Dataset.from_tensor_slices(render_poses)
    test_bds_ds = tf.data.Dataset.from_tensor_slices(tf.gather(bds, i_test))
    test_paths_ds = tf.data.Dataset.from_tensor_slices(tf.gather(img_paths, i_test))

    test_ds = (tf.data.Dataset.zip((test_paths_ds, test_poses_ds, test_render_poses, test_bds_ds))
               .repeat()
               .batch(len(i_test)) # Always batch entire test dataset
               .map(load_image)
               .prefetch(tf.data.experimental.AUTOTUNE)
    ) 

    val_poses_ds = tf.data.Dataset.from_tensor_slices(tf.gather(poses, i_val))
    val_bds_ds = tf.data.Dataset.from_tensor_slices(tf.gather(bds, i_val))
    val_paths_ds = tf.data.Dataset.from_tensor_slices(tf.gather(img_paths, i_val))

    val_ds = (tf.data.Dataset.zip((val_paths_ds, val_poses_ds, val_bds_ds))
               .batch(max_images)
               .repeat()
               .map(random_sample_1d)
               .unbatch()
               .batch(1) # Always sample 1 image for validation
               .map(load_image)
               .prefetch(tf.data.experimental.AUTOTUNE)
    ) 
    
    return train_ds, test_ds, val_ds

def get_num_task_datasets(tasks_dir):
    task_paths = glob.glob(os.path.join(tasks_dir, '*/'))
    num_tasks = len(task_paths)
    return num_tasks


def get_dataset_of_tasks(tasks_dir, meta_batch_size=6, task_batch_size=1024, max_images_per_task=10000):

    task_paths = glob.glob(os.path.join(tasks_dir, '*/'))
    num_tasks = len(task_paths)
    dataset_iters = []
    for task_path in task_paths:
        dataset_iters.append(iter(get_dataset_for_task(task_path, batch_size=task_batch_size, max_images=max_images_per_task)))
    
    def next_elem_from_dataset(i):
        i = tf.convert_to_tensor(i)
        def _numpy_wrapper(i):
            _next = next(dataset_iters[i])
            # print(f'Read from dataset {i}')
            return _next
        _next = tuple(tf.numpy_function(_numpy_wrapper, [i], (tf.float32, tf.float32, tf.float32)))
        return _next
    
    ds = (tf.data.Dataset.range(num_tasks)
          .repeat()
          .map(next_elem_from_dataset)
          .batch(meta_batch_size)
          .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return ds


# ds = get_dataset_for_task('E:/tanks-and-temples/image_sets/Ballroom/')
# # ds = get_dataset_of_tasks('E:/nerf/data/tanks_and_temples')

# before = time.time()
# for x in ds:
#     # print(x.shape)

#     after = time.time()
#     dt = after - before
#     print(dt)

#     # time.sleep(3)

#     before = time.time()
#     # plt.imshow(x)
#     # plt.show()
