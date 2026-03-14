from layers import disp_to_depth
import networks
import cv2
import os
import torch
import scipy.misc
from scipy import io
import numpy as np
from get_monovit import get_monovit_pretrained

load_weights_folder = r'/home/lmq/ZS/EmoDepth/my_logs/mono_model_100/models/weights_19'
encoder = networks.emo_encoder.emo_small(r'/home/lmq/ZS/EmoDepth/pretrained')
depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

monovit = get_monovit_pretrained(640)

# monovit_model = get_monovit_pretrained(640)
# encoder = monovit_model.encoder
# depth_decoder = monovit_model.decoder

encoder_path = os.path.join(load_weights_folder, "encoder.pth")
decoder_path = os.path.join(load_weights_folder, "depth.pth")

encoder_dict = torch.load(encoder_path)
#
model_dict = encoder.state_dict()
encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
depth_decoder.load_state_dict(torch.load(decoder_path))

# model_dict = encoder.state_dict()
# encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
# depth_decoder.load_state_dict(torch.load(decoder_path))
#
encoder.cuda(0)
encoder.eval()
depth_decoder.cuda(0)
depth_decoder.eval()

monovit.cuda(0)
monovit.eval()
main_path = r"/home/lmq/ZS/EmoDepth/make3D"


def compute_errors(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log


with open(os.path.join(main_path, "make3d_test_files.txt")) as f:
    test_filenames = f.read().splitlines()
test_filenames = map(lambda x: x[4:-4], test_filenames)

depths_gt = []
images = []
ratio = 2
h_ratio = 1 / (1.33333 * ratio)
color_new_height = 1704 // 2
depth_new_height = 21
for filename in test_filenames:
    mat = scipy.io.loadmat(os.path.join(main_path, "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)))
    depths_gt.append(mat["Position3DGrid"][:, :, 3])

    image = cv2.imread(os.path.join(main_path, "Test134", "img-{}.jpg".format(filename)))
    image = image[(2272 - color_new_height) // 2:(2272 + color_new_height) // 2, :]
    images.append(image[:, :, ::-1])
    # cv2.imwrite(os.path.join(main_path, "Test134_cropped", "img-{}.jpg".format(filename)), image)
depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt)
depths_gt_cropped = map(lambda x: x[(55 - 21) // 2:(55 + 21) // 2], depths_gt)

# pred_disps = np.load(path_to_pred_disps)
#
# errors = []
# for i in range(len(test_filenames)):
#     depth_gt = depths_gt_cropped[i]
#     depth_pred = 1 / pred_disps[i]
#     depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
#     mask = np.logical_and(depth_gt > 0, depth_gt < 70)
#     depth_gt = depth_gt[mask]
#     depth_pred = depth_pred[mask]
#     depth_pred *= np.median(depth_gt) / np.median(depth_pred)
#     depth_pred[depth_pred > 70] = 70
#     errors.append(compute_errors(depth_gt, depth_pred))
# mean_errors = np.mean(errors, 0)

depths_gt_cropped = list(depths_gt_cropped)
errors = []
with torch.no_grad():
    for i in range(len( images)):
        input_color = images[i]
        input_color =  cv2.resize(input_color/255.0, (640, 192), interpolation=cv2.INTER_NEAREST)#<----1
        input_color = torch.tensor(input_color, dtype = torch.float).permute(2,0,1)[None,:,:,:]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 假设input_color是你的输入数据
        input_color = input_color.to(device)

        features = encoder(input_color)

        output = depth_decoder(features)

        pred_disp,_ = disp_to_depth(output[("disp", 0)], 0.01, 100) #<---2
        pred_disp = pred_disp.squeeze().cpu().numpy()
        depth_gt = depths_gt_cropped[i]
        depth_pred = 1 / pred_disp
        depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
        mask = np.logical_and(depth_gt > 0, depth_gt < 70)
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= np.median(depth_gt) / np.median(depth_pred)
        depth_pred[depth_pred > 70] = 70
        errors.append(compute_errors(depth_gt, depth_pred))
    mean_errors = np.mean(errors, 0)



print(("{:>8} | " * 4).format("abs_rel", "sq_rel", "rmse", "rmse_log"))
print(("{: 8.3f} , " * 4).format(*mean_errors.tolist()))
