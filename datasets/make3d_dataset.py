import os
import PIL.Image as pil
import numpy as np

from .mono_dataset import MonoDataset


class MAKE3DDataset(MonoDataset):
    """Data loader for the Make3D dataset"""

    def __init__(self, data_path, filenames, height, width, frame_idxs, num_scales, is_train=False, img_ext='.jpg'):
        super(MAKE3DDataset, self).__init__(data_path, filenames, height, width, frame_idxs, num_scales, is_train, img_ext)
        self.side_map = {"l": "image", "r": "image"}  # Make3D does not have stereo, we use "image" placeholder

    def get_image_path(self, folder, frame_index, side):
        # The folder is typically an empty string ""
        f_str = "{:06d}{}".format(int(frame_index), self.img_ext)
        return os.path.join(self.data_path, "images", f_str)

    def get_depth(self, folder, frame_index, side, do_flip):
        # Ground truth depth maps should be in self.data_path/depths/xxxxx.png
        depth_path = os.path.join(self.data_path, "depths", "{:06d}.png".format(int(frame_index)))

        depth_gt = pil.open(depth_path)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256.0  # Assume KITTI format (16-bit PNG)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
