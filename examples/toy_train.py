import os
import math
import torch
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm


def get_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        if os.path.exists(f"{self.dataset_name}.bin"):
            self.tokens = torch.load(f"{self.dataset_name}.bin")
        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, f"{self.dataset_name}.bin")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py

# Create compiled and non-compiled versions
def _zeropower_via_newtonschulz5_impl(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Use bfloat16 on CUDA, but float32 on MPS/CPU for compatibility
    if G.device.type == 'cuda':
        X = G.bfloat16()
    else:
        X = G.float()
    
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X

# Create compiled version for CUDA/CPU
_zeropower_compiled = torch.compile(_zeropower_via_newtonschulz5_impl)

def zeropower_via_newtonschulz5(G, steps):
    """
    Conditionally compiled Newton-Schulz function.
    Uses torch.compile on CUDA/CPU for performance, but skips on MPS for compatibility.
    """
    if G.device.type == 'mps':
        # Use non-compiled version on MPS to avoid compatibility issues
        return _zeropower_via_newtonschulz5_impl(G, steps)
    else:
        # Use compiled version on CUDA/CPU for better performance
        return _zeropower_compiled(G, steps)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


class MORGN(torch.optim.Optimizer):
    """
    MORGN - Momentum Orthogonalization via Recursive Gauss-Newton.

    Implements a left preconditioned update for 2D parameters using a
    Recursive Gauss-Newton (RLS-style) inverse Hessian approximation P.

    For each matrix parameter W (m x n), we maintain P (m x m) on the
    smaller/left dimension (transposing if m > n). The update is

        U = P @ G   (left Newton step on gradient G = dL/dW)
        W <- W - lr * U

    P is updated by a rank-1 RLS/Sherman-Morrison formula using a small
    set of gradient directions (columns of G) per step. This captures up
    to K directions per iteration at O(m^2 K) cost.

    Arguments:
        morgn_params: iterable of 2D parameters updated by MORGN
        lr: base learning rate
        wd: weight decay (L2) applied like AdamW on W
        lambda_: forgetting factor in (0,1], closer to 1 retains history
        eps: initial diagonal of P0 = (1/eps) * I to start well-conditioned
        directions: number of gradient columns to assimilate per step
        adamw_params: parameters optimized by AdamW fallback
        adamw_betas, adamw_eps: AdamW settings for fallback
    """

    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.0,
        morgn_params=None,
        lambda_: float = 0.99,
        eps: float = 1e-3,
        directions: int = 8,
        step_clamp: float = 0.0,
        adamw_params=None,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        if morgn_params is None:
            morgn_params = []
        if adamw_params is None:
            adamw_params = []

        defaults = dict(
            lr=lr,
            wd=wd,
            lambda_=lambda_,
            eps=eps,
            directions=directions,
            step_clamp=step_clamp,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(morgn_params) + list(adamw_params)
        super().__init__(params, defaults)

        # Tag parameters and initialize state
        for p in morgn_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_morgn"] = True
            m, n = p.shape
            if m <= n:
                left_dim = m
                self.state[p]["transposed"] = False
            else:
                left_dim = n
                self.state[p]["transposed"] = True  # operate on p.T
            device = p.device
            dtype = p.dtype
            # P0 = (1/eps) * I (large to allow quick adaptation)
            eps_init = self.defaults["eps"]
            P0 = torch.eye(left_dim, device=device, dtype=dtype) / eps_init
            self.state[p]["P"] = P0

        for p in adamw_params:
            self.state[p]["use_morgn"] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            lambda_ = group["lambda_"]
            directions = int(group["directions"])
            step_clamp = float(group.get("step_clamp", 0.0))
            beta1, beta2 = group.get("adamw_betas", (0.9, 0.95))
            eps = group.get("adamw_eps", 1e-8)

            # MORGN updates
            for p in [pp for pp in group["params"] if self.state[pp].get("use_morgn", False)]:
                g = p.grad
                if g is None:
                    continue

                st = self.state[p]
                P = st["P"]
                transposed = st["transposed"]

                # Shape to left form (m x k)
                G = g.T if transposed else g
                m, k = G.shape

                # Preconditioned left Newton step
                # U_left = P @ G, then map back
                U_left = P @ G
                update = U_left.T if transposed else U_left # latest gradient times previous inverse Hessian

                # Weight decay like AdamW
                if wd != 0.0:
                    p.data.mul_(1 - lr * wd)

                # Step clamp by Frobenius norm, relative to parameter norm
                if step_clamp > 0.0:
                    upd_norm = torch.norm(update, p='fro')
                    ref = torch.norm(p.data, p='fro') + 1e-12
                    max_upd = step_clamp * ref
                    if torch.isfinite(upd_norm) and upd_norm > max_upd:
                        scale = (max_upd / (upd_norm + 1e-12))
                        update = update * scale

                p.data.add_(update, alpha=-lr)

                # Update P using Sherman-Morrison formula for all gradient columns
                # For each column v in G, we update H_inv where H gets rank-1 update v*v^T
                # Sherman-Morrison: (H + v*v^T)^{-1} = H^{-1} - (H^{-1}*v*v^T*H^{-1}) / (1 + v^T*H^{-1}*v)
                with torch.no_grad():
                    # Vectorized Woodbury update across all k columns using precomputed U_left = P @ G
                    gn_mat = U_left  # shape (m, k)
                    # Compute S = I_k + G^T @ (P @ G) and update without explicit inverse for stability
                    if gn_mat.dtype in (torch.float16, torch.bfloat16):
                        gn32 = gn_mat.float()
                        G32 = G.float()
                        I_k = torch.eye(k, device=gn32.device, dtype=gn32.dtype)
                        S = I_k + G32.T @ gn32
                        S = S + (1e-12 * I_k)
                        if S.device.type == 'mps':
                            # Try explicit inverse on MPS; fall back to CPU solve if unsupported
                            try:
                                S_inv = torch.linalg.inv(S)
                                X = S_inv @ gn32.T  # (k, m)
                            except Exception:
                                X_cpu = torch.linalg.solve(S.cpu(), gn32.T.cpu())
                                X = X_cpu.to(gn32.device)
                        else:
                            X = torch.linalg.solve(S, gn32.T)  # (k, m)
                        delta = (gn32 @ X).to(gn_mat.dtype)
                    else:
                        I_k = torch.eye(k, device=gn_mat.device, dtype=gn_mat.dtype)
                        S = I_k + G.T @ gn_mat
                        S = S + (gn_mat.new_tensor(1e-12) * I_k)
                        if S.device.type == 'mps':
                            # Try explicit inverse on MPS; fall back to CPU solve if unsupported
                            try:
                                S_inv = torch.linalg.inv(S)
                                X = S_inv @ gn_mat.T  # (k, m)
                            except Exception:
                                X_cpu = torch.linalg.solve(S.cpu(), gn_mat.T.cpu())
                                X = X_cpu.to(gn_mat.device)
                        else:
                            X = torch.linalg.solve(S, gn_mat.T)  # (k, m)
                        delta = gn_mat @ X
                    P.sub_(delta)
                    
                    # Apply forgetting factor (exponential decay of old information)
                    P.mul_(1.0 / lambda_)
                    
                    # Keep symmetry (P should be symmetric)
                    P.copy_(0.5 * (P + P.T))
                    st["P"] = P

            # AdamW fallback for non-2D or excluded params
            for p in [pp for pp in group["params"] if not self.state[pp].get("use_morgn", False)]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(p)
                    state["moment2"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                m1 = state["moment1"]
                m2 = state["moment2"]
                m1.lerp_(g, 1 - beta1)
                m2.lerp_(g.square(), 1 - beta2)
                g_hat = m1 / (eps + m2.sqrt())
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                if wd != 0.0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(g_hat, alpha=-lr * (bias_correction2 ** 0.5) / bias_correction1)

        return loss


def get_model_and_dataloader(model_name, dataset_name, hidden_size):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    train_dataset = load_dataset(name2path[dataset_name], trust_remote_code=True)
    if model_name == "qwen":
        tokenizer = Qwen2Tokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )
    else:
        assert 0, f"model {model_name} not supported"
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    if model_name == "qwen":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=513,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=16,
            num_hidden_layers=12,
            num_key_value_heads=16,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
        model = Qwen2ForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"
    return model, train_loader


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    elif optimizer_name == "morgn":
        morgn_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return MORGN(
            lr=lr,
            wd=wd,
            morgn_params=morgn_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, f"optimizer {optimizer_name} not supported"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=1024)
    args = parser.parse_args()
    logger.add(f"logs/train_{args.model}_{args.optimizer}_lr{args.lr}.log")

    model, train_loader = get_model_and_dataloader(
        args.model, args.dataset, args.hidden_size
    )
    optimizer = get_optimizer(
        args.optimizer, model, lr=args.lr
    )

    device = get_device()
    model.to(device)

    model.train()
    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )
    for epoch in range(epoch):
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            input_ids = batch
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            logger.info(
                f"Epoch: {epoch} Step: {step} LR: {optimizer.param_groups[0]['lr']} Training loss: {loss.item()}"
            )
