import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

'''
这段 AutoBlurModule 代码实现了一个自适应模糊模块，核心思想是：只对图像中高频区域进行高斯模糊，其余部分保持原样，以保留细节、同时降低噪点或伪影。
下面我将逐行为你解释，并指出其优点、应用场景及快速学习要点。
图像去伪影、降噪（只处理纹理区域）
超分辨、图像增强前的预处理
视频帧插值前的稳定区域保持

它的目标是：
    仅对图像中的“高频区域”进行模糊处理，其余部分保持清晰。
这意味着：
    边缘、纹理、细节等处如果太过锐利、可能带来伪影的部分 → 被轻度模糊
    背景、大面积平滑区域 → 完全保留原图清晰度

| 对比项                 | 使用前（原图）      | 使用后（AutoBlur）   |
| ------------------- | ------------ | --------------- |
| **边缘细节**            | 锐利，可能有锯齿或噪声  | 柔和平滑，更自然        |
| **背景区域**            | 原始状态，有可能过度锐化 | 几乎不变，仍然清晰       |
| **高频区域（如纹理、头发、草地）** | 可能出现噪声/压缩伪影  | 细节依然保留，但减少“刺眼感” |
| **整体观感**            | 尖锐、有瑕疵的细节    | 更协调、柔和、视觉舒适     |


| 概念              | 解释                   |
| --------------- | -------------------- |
| **高频区域**        | 图像变化剧烈的部分，通常是边缘或细节区域 |
| **梯度计算**        | 通过图像差分衡量局部变化程度       |
| **自适应模糊**       | 只在需要模糊的区域应用高斯滤波      |
| **AvgPool 感受野** | 用于局部区域统计判断像素是否处在高频区域 |

'''
class AutoBlurModule(nn.Module):
    def __init__(self, receptive_field_of_hf_area,
                 hf_pixel_thresh=0.2,
                 hf_area_percent_thresh=60,
                 gaussian_blur_kernel_size=11,
                 gaussian_blur_sigma=5.0,
                 ):
        super(AutoBlurModule, self).__init__()

        self.receptive_field_of_hf_area = receptive_field_of_hf_area
        self.hf_pixel_thresh = hf_pixel_thresh
        self.hf_area_ratio = hf_area_percent_thresh / 100

        self.gaussian_blur = transforms.GaussianBlur(gaussian_blur_kernel_size,
                                                     sigma=gaussian_blur_sigma)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=receptive_field_of_hf_area, stride=1,
            padding=(receptive_field_of_hf_area - 1) // 2)

    @staticmethod
    def compute_spatial_grad(ipt):
        grad_u = torch.abs(ipt[:, :, :, :-1] - ipt[:, :, :, 1:]).sum(1, True)
        grad_v = torch.abs(ipt[:, :, :-1, :] - ipt[:, :, 1:, :]).sum(1, True)

        grad_u = F.pad(grad_u, (0, 1))
        grad_v = F.pad(grad_v, (0, 0, 0, 1))

        grad_l2_norm = torch.sqrt(grad_u ** 2 + grad_v ** 2)
        return grad_l2_norm

    def forward(self, raw_img):
        # Gaussian blur the whole image first.
        blurred_img = self.gaussian_blur(raw_img)

        # Whether it is a high frequency pixel.
        spatial_grad = self.compute_spatial_grad(raw_img)
        is_hf_pixel = spatial_grad > self.hf_pixel_thresh

        # Compute how many high frequency pixels are around.
        avg_pool_freq = self.avg_pool(is_hf_pixel.float())

        # If 60% of the surrounding pixels are high frequency,
        # the pixel considered to be in the high frequency region.
        is_in_hf_area = avg_pool_freq > self.hf_area_ratio

        weight_blur = avg_pool_freq * is_in_hf_area

        # Only pixels located in high frequency regions are
        # gaussian blurred, with other pixels unchanged.
        # The more the avg freq, the more the pixel is blurred.
        auto_blurred = blurred_img * weight_blur + \
                       raw_img * (1 - weight_blur)

        return auto_blurred