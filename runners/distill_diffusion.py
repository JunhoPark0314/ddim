import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from functions.denoising import compute_alpha

from models.diffusion import Encoder, Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
from copy import deepcopy


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class DistillDiffusion(object):
    def __init__(self, args, config, tb_logger, device=None):
        self.args = args
        self.tb_logger = tb_logger
        self.trg_config = config
        # TODO : hard codded cond channel
        cond_channel = 128
        self.config = deepcopy(config)
        self.enc_config = deepcopy(config)
        self.config.model.use_cond = True
        self.config.model.cond_channel = cond_channel
        self.config.model.type = 'cond_simple'
        self.enc_config.model.out_ch = cond_channel

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        trg_config, enc_config = self.trg_config, self.enc_config
        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        trg_model = Model(trg_config)
        trg_model = trg_model.to(self.device)
        trg_model = torch.nn.DataParallel(trg_model)
        trg_model = trg_model.eval()

        encoder = Encoder(enc_config)
        encoder = encoder.to(self.device)
        encoder = torch.nn.DataParallel(encoder)

        optimizer = get_optimizer(self.config, model.parameters())
        enc_optimizer = get_optimizer(self.enc_config, encoder.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            enc_ema_helper = EMAHelper(mu=self.enc_config.model.ema_rate)
            ema_helper.register(model)
            enc_ema_helper.register(encoder)
        else:
            ema_helper = None

        trg_states = torch.load(os.path.join(self.args.trg_path, "ckpt.pth"))
        trg_model.load_state_dict(trg_states[0])
        model.load_state_dict(trg_states[0], strict=False)

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])

            encoder.load_state_dict(states[2])
            states[3]["param_groups"][0]["eps"] = self.enc_config.optim.eps
            enc_optimizer.load_state_dict(states[3])

            start_epoch = states[4]
            step = states[5]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[6])
                enc_ema_helper.load_state_dict(states[7])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e1 = torch.randn_like(x)
                e2 = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.args.timesteps-1, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.args.timesteps - t - 2], dim=0)[:n] + 1
                trg_step = self.num_timesteps // self.args.timesteps
                t *= trg_step
                
                loss = loss_registry[config.model.type](model, trg_model, encoder, trg_step, x, t, e1, e2, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                enc_optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                    torch.nn.utils.clip_grad_norm_(
                        encoder.parameters(), enc_config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()
                enc_optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)
                    enc_ema_helper.update(encoder)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        encoder.state_dict(),
                        enc_optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                        states.append(enc_ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                
                if step % self.config.training.sample_freq == 0 or step == 1:
                    self.sample_sequence(model, trg_model, encoder, x, step)

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model, trg_model, encoder, sample_x, step=0):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        sample_x = sample_x[:8]

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x, trg_x0, cond_x0, org_x0 = self.sample_image(x, model, trg_model, encoder, sample_x, last=False)

        x = [inverse_data_transform(config, y) for y in x]
        trg_x0 = [inverse_data_transform(config, y) for y in trg_x0][::-1]
        cond_x0 = [inverse_data_transform(config, y) for y in cond_x0][::-1]
        org_x0 = [inverse_data_transform(config, y) for y in org_x0]
        N = len(x)

        x = torch.cat(x).reshape(N, 8, config.data.channels, config.data.image_size, config.data.image_size).permute(1,0,2,3,4).flatten(0,1)
        trg_x0 =  torch.cat(trg_x0).reshape(N, 8, config.data.channels, config.data.image_size, config.data.image_size).permute(1,0,2,3,4).flatten(0,1) 
        cond_x0 =  torch.cat(cond_x0).reshape(N, 8, config.data.channels, config.data.image_size, config.data.image_size).permute(1,0,2,3,4).flatten(0,1) 
        org_x0 =  torch.cat(org_x0).reshape(N, 8, config.data.channels, config.data.image_size, config.data.image_size).permute(1,0,2,3,4).flatten(0,1) 

        os.makedirs(os.path.join(self.args.exp, self.args.image_folder, 'sample'), exist_ok=True)
        tvu.save_image(tvu.make_grid(x), os.path.join(self.args.exp, self.args.image_folder, 'sample', f"seq_{step}.png"))

        os.makedirs(os.path.join(self.args.exp, self.args.image_folder, 'trg'), exist_ok=True)
        tvu.save_image(tvu.make_grid(trg_x0), os.path.join(self.args.exp, self.args.image_folder, 'trg', f"seq_{step}.png"))

        os.makedirs(os.path.join(self.args.exp, self.args.image_folder, 'cond'), exist_ok=True)
        tvu.save_image(tvu.make_grid(cond_x0), os.path.join(self.args.exp, self.args.image_folder, 'cond', f"seq_{step}.png"))

        os.makedirs(os.path.join(self.args.exp, self.args.image_folder, 'org'), exist_ok=True)
        tvu.save_image(tvu.make_grid(org_x0), os.path.join(self.args.exp, self.args.image_folder, 'org', f"seq_{step}.png"))

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, trg_model, encoder, sample_x, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        trg_x0_list = []
        cond_x0_list = []
        with torch.no_grad():
            assert x.shape == sample_x.shape
            xt_list, t_list, a_list = self.sample_noised_image(sample_x, self.betas, self.args.timesteps)
            trg_step = self.num_timesteps // self.args.timesteps
            cond = []
            for i, (xt, t, a) in enumerate(zip(xt_list, t_list, a_list)):
                eps_pred = trg_model(xt, t)
                x0 = (xt - (1-a).sqrt() * eps_pred) / a.sqrt()
                # TODO: Assert t is trg_t
                trg_x0_list.append(x0)
                cond.append(encoder(x0, t))
                if i < len(xt_list)-1:
                    eps_cond = model(xt_list[i+1], t_list[i+1], cond[-1])
                    x0_cond = (xt_list[i+1] - (1 - a_list[i+1]).sqrt() * eps_cond) / (a_list[i+1]).sqrt()
                    cond_x0_list.append(x0_cond)

            cond = cond[:-1]
            trg_x0_list = trg_x0_list[:-1]

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, cond=cond, eta=self.args.eta)
            org_xs = generalized_steps(x, seq, trg_model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x + (trg_x0_list, cond_x0_list, org_xs[1])

    def sample_noised_image(self, x, b, timesteps):
        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // timesteps
            seq = list(range(0, self.num_timesteps, skip))
        elif self.args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(self.num_timesteps * 0.8), timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        
        noised_x = []
        t_list = []
        a_list = []

        for t in seq:
            e = torch.randn_like(x)
            t = torch.ones(x.size(0), device=x.device) * t
            a = compute_alpha(b, t.long())
            noised_x.append(x * a.sqrt() + e * (1-a).sqrt())
            t_list.append(t)
            a_list.append(a)
        
        noised_x = [noised_x[0]] + noised_x
        t_list = [t_list[0]] + t_list
        a_list = [a_list[0]] + a_list

        return noised_x, t_list, a_list

    def test(self):
        pass
