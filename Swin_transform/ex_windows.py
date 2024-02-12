
import torch
import torchvision
from  torch import nn
from  torch.utils.data import DataLoader

class SwinTransformerBlock(nn.Module):
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x   # x = [2,3136,96]
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # (2, 56, 56, 96)

        # cyclic shift
        # 在第一次我们是W-MSA，没有滑动窗口，所以self.shift_size > 0 =False
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows （把attention后的数据还原成原来输入的shape）
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            print(x.shape)
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        print(x.shape)

        # FFN
        x = shortcut + self.drop_path(x)
        print(x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        print(x.shape)
        return x
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)  # (2,8,7,8,7,96):指把56*56的patch按照7*7的窗口划分
    print(x.shape)  # (2,8,7,8,7,96)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) # window的数量 H/7 * W/7 *batch
    print(windows.shape)
    # windows=(128, 7, 7, 96)
    # 128 = batch_size * 8 * 8 = 128窗口的数量
    # 7 = window_size 窗口的大小尺寸，说明每个窗口包含49个patch
    return windows
