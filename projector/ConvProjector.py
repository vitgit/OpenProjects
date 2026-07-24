import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import models


#### Projectors ############################################ Begin
class ConvProjectorQQinv(nn.Module):
    def __init__(
            self,
            C,
            Dc=64,
            kernel_size_Q=3,
            kernel_size_QQ_inv=5
    ):
        super().__init__()
        padding_Q = kernel_size_Q // 2
        padding_QQ_inv = kernel_size_QQ_inv // 2

        # Q*: C -> Dc
        self.Q_star = nn.Conv2d(C, Dc, kernel_size=kernel_size_Q, padding=padding_Q, bias=False)

        # (Q*Q)^(-1) approx: Dc -> Dc
        self.QQ_inv = nn.Conv2d(Dc, Dc, kernel_size=kernel_size_QQ_inv, padding=padding_QQ_inv, groups=Dc, bias=False)

        # Q: Dc -> C
        self.Q = nn.Conv2d(Dc, C, kernel_size=kernel_size_Q, padding=padding_Q, bias=False)

    def forward(self, x, return_all=False):
        x_in = x

        z = self.Q_star(x_in)     # [B, Dc, H, W]
        z_inv = self.QQ_inv(z)    # [B, Dc, H, W]
        x_proj = self.Q(z_inv)    # [B, C, H, W]

        if return_all:
            return x_proj, {
                "x_in": x_in,
                "z_Q_star": z,
                "z_QQ_inv": z_inv,
            }

        return x_proj

class ConvProjectorExactInverse(nn.Module):

    def __init__(
            self,
            C,
            stride=2,
            kernel_size_QQ_inv=3,
            eps=1e-3,
            local_coarse_smoothing = False,
    ):
        super().__init__()

        self.C = C
        self.stride = stride
        self.eps = eps
        self.local_coarse_smoothing = local_coarse_smoothing
        self.Ainv = None

        # ---------------------------------
        # Restriction
        # ---------------------------------


        if local_coarse_smoothing:
            # ---------------------------------
            # Optional local coarse smoothing
            # ---------------------------------
            padding_QQ = kernel_size_QQ_inv // 2

            self.QQ_local = nn.Conv2d(
                in_channels=C,
                out_channels=C,
                kernel_size=kernel_size_QQ_inv,
                padding=padding_QQ,
                groups=C,  # if groups=C, each channel is convoluted independently
                bias=False
            )

        # ---------------------------------
        # Generalized transposed conv params
        # ---------------------------------
        kernel_size_T = 2 * stride
        padding_T = stride // 2

        # Exact output size correction
        output_padding = stride % 2

        # ---------------------------------
        # Prolongation
        # ---------------------------------
        self.Q = nn.ConvTranspose2d(
            in_channels=C,
            out_channels=C,
            kernel_size=kernel_size_T,
            stride=stride,
            padding=padding_T,
            output_padding=output_padding,
            groups=C,
            bias=False
        )

    def apply_Q_star(self, x):
        return F.conv2d(
            x,
            weight=self.Q.weight,
            stride=self.stride,
            padding=self.Q.padding[0],
            groups=self.C
        )

    def build_QtQ_inverse(self, Hc, Wc, device):
        Nc = Hc * Wc
        Ainv = []

        for c in range(self.C):
            Qmat = torch.zeros(Nc, Nc, device=device)

            for i in range(Nc):
                basis = torch.zeros(1, self.C, Hc, Wc, device=device)
                basis[0, c].view(-1)[i] = 1.0

                out = self.Q(basis)
                out = self.apply_Q_star(out)

                Qmat[:, i] = out[0, c].reshape(-1)

            #--------------------------------
            symmetry_err = (Qmat - Qmat.T).abs().max()
            assert symmetry_err < 1e-5, "Q*Q is not symmetric"
            #--------------------------------

            A = Qmat + self.eps * torch.eye(Nc, device=device)

            # print(f"channel {c}, cond(Q*Q + eps I):", torch.linalg.cond(A))

            Ainv.append(torch.linalg.inv(A))

        self.Ainv = torch.stack(Ainv, dim=0)  # [C, Nc, Nc]

    def forward(self, x, return_all=False):

        x_in = x

        # ---------------------------------
        # Restriction
        # ---------------------------------
        z = self.apply_Q_star(x_in)

        if self.local_coarse_smoothing:
            # Optional local smoothing
            z = self.QQ_local(z)

        B, C, Hc, Wc = z.shape

        # ---------------------------------
        # Flatten spatial dims
        # ---------------------------------
        z_flat = z.reshape(B, C, -1)

        if self.training:
            self.build_QtQ_inverse(Hc, Wc, x.device)
        elif self.Ainv is None:
            self.build_QtQ_inverse(Hc, Wc, x.device)

        u = torch.einsum("cij,bcj->bci", self.Ainv, z_flat)


        # ---------------------------------
        # Restore coarse grid
        # ---------------------------------
        z_inv = u.reshape(B, C, Hc, Wc)

        # ---------------------------------
        # Prolongation
        # ---------------------------------
        x_proj = self.Q(z_inv)

        assert x_proj.shape[-2:] == x_in.shape[-2:]

        if return_all:
            return x_proj, {
                "x_in": x_in,
                "z_Q_star": z,
                "z_QQ_inv": z_inv,
                "x_proj": x_proj,
            }

        return x_proj

#### Projectors ############################################ End

class ResNetWithProjectorConv(nn.Module):
    def __init__(
            self,
            projector_name='ConvProjectorQQinv', # 'ConvProjectorExactInverse'
            use_projector=True,
            #
            Dc=16,      # in ConvProjectorQQinv. The number of coarse latent channels.
            # or #
            stride=2,   # in ConvProjectorExactInverse. The size of aggregation.
            #
            alpha_const=None,
            kernel_size_Q=3,
            kernel_size_QQ_inv=5):
        super().__init__()

        self.use_projector  = use_projector
        self.alpha_const    = alpha_const
        self.current_alpha  = None
        self.plot_image     = False

        backbone = models.resnet18(weights=None)

        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.classifier = nn.Linear(512, 1)
        if use_projector:
            if projector_name == 'ConvProjectorQQinv':
                self.projector = ConvProjectorQQinv(
                    C=3,
                    Dc=Dc,
                    kernel_size_Q=kernel_size_Q,
                    kernel_size_QQ_inv=kernel_size_QQ_inv)
            elif projector_name == 'ConvProjectorExactInverse':
                self.projector = ConvProjectorExactInverse(
                    C=3,
                    stride=stride,
                    kernel_size_QQ_inv=kernel_size_QQ_inv)
            else:
                raise NotImplementedError('Unknown projector type')
            self.logit_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):

        proj_dict = x_proj = None
        if self.use_projector:
            if self.alpha_const is not None:
                alpha = self.alpha_const
                self.current_alpha = x.new_tensor(alpha)
                if alpha < 1.0:
                    x_proj, proj_dict = self.projector(x, return_all=True)
                    x_corrected = alpha * x + (1 - alpha) * x_proj
                else:
                    x_corrected = x
            else:
                x_proj, proj_dict = self.projector(x, return_all=True)
                alpha = torch.sigmoid(self.logit_alpha)
                self.current_alpha = alpha.detach()
                x_corrected = alpha * x + (1 - alpha) * x_proj
            #------------------------------------
            if self.plot_image and x_proj is not None:
                show_images(self.x_clean, x, x_proj, x_corrected, idx=0)

                # #---------------------------------------
                # blurred = F.avg_pool2d(x, 4, stride=1, padding=2)
                # show_images(self.x_clean, blurred, x_proj, x_corrected, idx=0)
                # #---------------------------------------

                if proj_dict is not None and x_proj is not None:
                    print("x:", proj_dict["x_in"].norm().item())
                    print("z:", proj_dict["z_Q_star"].norm().item())
                    print("z_inv:", proj_dict["z_QQ_inv"].norm().item())
                    print("x_proj:", x_proj.norm().item())

        #------------------------------------
        else:
            x_corrected = x

        # ---- backbone ----
        h = self.backbone(x_corrected)   # [B, 512, 4, 4]

        # ---- pooling ----
        h = F.adaptive_avg_pool2d(h, (1, 1)).reshape(h.size(0), -1)

        # ---- classification ----
        out = self.classifier(h)

        return out

def show_images(x_clean, x_noisy, x_proj, x_out, idx=0):

    def to_img(t, normalize=True):

        t = t[idx].detach().cpu()
        t = t.permute(1,2,0)

        if normalize:
            t_min = t.min()
            t_max = t.max()
            t = (t - t_min) / (t_max - t_min + 1e-8)
        else:
            t = torch.clamp(t, 0, 1)

        return t

    # ---- differences ----
    diff_proj = (x_proj - x_noisy).abs()
    diff_out  = (x_out  - x_noisy).abs()

    plt.figure(figsize=(18,6))

    plt.subplot(2,4,1)
    plt.title("Clean")
    plt.imshow(to_img(x_clean))

    plt.subplot(2,4,2)
    plt.title("Noisy")
    plt.imshow(to_img(x_noisy))

    plt.subplot(2,4,3)
    plt.title("Projected")
    plt.imshow(to_img(x_proj))

    plt.subplot(2,4,4)
    plt.title("Blended")
    plt.imshow(to_img(x_out))

    plt.subplot(2,4,5)
    plt.title("|Proj - Noisy|")
    plt.imshow(to_img(diff_proj, normalize=True))

    plt.subplot(2,4,6)
    plt.title("|Blend - Noisy|")
    plt.imshow(to_img(diff_out, normalize=True))

    plt.tight_layout()
    plt.show()