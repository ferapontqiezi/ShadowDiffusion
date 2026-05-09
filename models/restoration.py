import torch
import torch.nn as nn
import utils
import torchvision
import os


def collated_label_to_str(y):
    """Default DataLoader collate turns a batch of strings into a list."""
    if isinstance(y, (list, tuple)):
        if len(y) == 1:
            return str(y[0])
        return "_".join(str(item) for item in y)
    return str(y)


def output_save_name(args, index, image_id_str):
    fmt = getattr(args, "output_name_format", "id")
    if fmt == "sequential":
        return f"{index:04d}.png"
    safe = image_id_str.replace(os.sep, "_").replace("/", "_")
    for ch in '\\:*?"<>|':
        safe = safe.replace(ch, "_")
    return f"{safe}_output.png"


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='snow', r=None):
        image_folder = os.path.join(self.args.image_folder, validation)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                y_str = collated_label_to_str(y)
                print(f"starting processing from image {y_str}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond, r=r)
                x_output = inverse_data_transform(x_output)
                name = output_save_name(self.args, i, y_str)
                utils.logging.save_image(x_output, os.path.join(image_folder, name))

    def diffusive_restoration(self, x_cond, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
    

class DiffusiveRestoration1:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration1, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume1):
            self.diffusion.load_ddm_ckpt(args.resume1, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='snow', r=None):
        image_folder = os.path.join(self.args.image_folder, validation, "stage1")
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                y_str = collated_label_to_str(y)
                print(f"starting processing from image {y_str}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond, r=r)
                x_output = inverse_data_transform(x_output)
                name = output_save_name(self.args, i, y_str)
                utils.logging.save_image(x_output, os.path.join(image_folder, name))

    def diffusive_restoration(self, x_cond, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list


class DiffusiveRestoration2:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration2, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume2):
            self.diffusion.load_ddm_ckpt(args.resume2, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='snow', r=None):
        image_folder = os.path.join(self.args.image_folder, validation, "stage2")
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                y_str = collated_label_to_str(y)
                print(f"starting processing from image {y_str}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond, r=r)
                x_output = inverse_data_transform(x_output)
                name = output_save_name(self.args, i, y_str)
                utils.logging.save_image(x_output, os.path.join(image_folder, name))

    def diffusive_restoration(self, x_cond, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list

