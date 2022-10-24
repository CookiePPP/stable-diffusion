import traceback
from datetime import timedelta

import torch
from torch import nn


#@torch.jit.script
def low_precision_to(x: torch.FloatTensor, weighted_random_rounding: bool = True):
    """
    Convert tensor to uint8 with dynamic range.
    Weighted random rounding allows more precision when averaging the values
    of non-changing elements across many iterations.
    """
    x_max = x.max()
    x_min = x.min()
    x_range = x_max - x_min
    # map [x_min, x_max] to [0, 255]
    x = (x - x_min) * (255/x_range)
    if weighted_random_rounding:
        # weighted random rounding, 0.2 has 80% chance to round to 0, 20% chance to round to 1
        x = torch.floor(x + torch.rand_like(x))
    else:
        # round to nearest integer
        x = torch.round(x)
    # convert to uint8
    x = x.to(torch.uint8)
    return x, x_min, x_max


#@torch.jit.script
def low_precision_from(x: torch.ByteTensor, x_min: torch.FloatTensor, x_max: torch.FloatTensor):
    """Convert tensor from uint8 with dynamic range to float32."""
    x_range = x_max - x_min
    x = x.to(torch.float32) * (x_range/ 255) + x_min
    return x

class LitEma(nn.Module):
    def __init__(self, model, rank=None,
            half_life=1000, step_interval: int = 10, warmup_steps=100,
            device='cpu', force_dtype=torch.float32, low_precision_transfer=False, rank_zero_only=True):
        super().__init__()
        self.rank_zero_only = rank_zero_only
        # only store shadow params on rank 0
        # send copy of shadow params to other ranks when requested
        
        self.rank = rank if rank is not None else model.global_rank
        self.half_life = half_life
        
        assert half_life > 0, 'half_life must be positive'
        # half_life is the amount of optimizer steps for shodow params to be 50% replaced by model params
        # Recommend setting half_life to half of the number of opt.step()'s in an epoch
        decay = 0.5 ** (step_interval/half_life)

        # step_interval: 1 = every iteration, 2 = every other iteration, etc.
        #              : Model ensembles work with as little as 8 total steps,
        #              : so feel free to set this to a high value like 32.
        self.interval = step_interval
        self.step_counter = 0
        
        self.device = device
        self.force_dtype = force_dtype
        self.low_precision_transfer = low_precision_transfer # not used right now # transfer shadow params to model params in low precision (fp16)
        
        # remove as '.' character is not allowed in buffers
        self.sname_lookup = {name: name.replace('.', '') for name, p in model.named_parameters() if p.requires_grad}
        
        self.collected_params = []
        
        # init tensors (if rank_zero or not rank_zero_only)
        self.init_shadow_params(model)
        self.decay = torch.tensor(decay, dtype=torch.float32)
        self.num_updates = torch.tensor(-warmup_steps//step_interval, dtype=torch.int)

    @torch.no_grad()
    def init_shadow_params(self, model):
        """
        Initialize the shadow parameters with the current parameters of the model.
        Since pytorch lightning doesn't initialize distributed world until after
        the model is created, we need to initialize the shadow params here,
        then delete them if they are not needed after initialization.
        """
        use_none = self.rank_zero_only and self.rank != 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.register_buffer(
                    self.sname_lookup[name],
                    self.detach_to(p).clone() if not use_none else None)
    
    @torch.no_grad()
    def del_redundant_params(self, model):
        """Delete any unneeded params from the shadow params."""
        use_none = self.rank_zero_only and self.rank != 0
        if not use_none:
            return
        for name, p in model.named_parameters():
            if p.requires_grad:
                setattr(self, self.sname_lookup[name], None)
    
    def detach_to(self, x):
        x = x.detach().data.to(self.device)
        if self.force_dtype: x = x.to(self.force_dtype)
        return x
    
    def forward(self, model):
        """Update the shadow parameters with the current parameters of the model."""
        self.to(self.device)
        self.del_redundant_params(model)
        
        if self.rank_zero_only and self.rank != 0:
            return
        
        self.step_counter += 1
        if self.step_counter % self.interval != 0:
            return
        
        self.num_updates += 1
        
        # calculate mean of parameters for first 144 steps (when using half_life=1000)
        # after 144 steps, use exponential moving average instead of mean
        num_updates = max(self.num_updates.item(), 0)
        decay_initial = 1 - 1/(num_updates+1)
        decay = min(decay_initial, self.decay.item())
        # (this reduces the impact of the first few steps on the shadow params)
        
        with torch.no_grad():
            model_params  = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())
            
            for key in model_params:
                if not model_params[key].requires_grad:
                    assert key not in self.sname_lookup
                    continue
                
                shadow_name = self.sname_lookup[key]
                shadow_params[shadow_name] = self.detach_to(shadow_params[shadow_name])
                
                # transfer model param to shadow param device
                model_param_cpu = self.detach_to(model_params[key].half())
                # update shadow param with model param
                shadow_params[shadow_name].mul_(decay).add_(model_param_cpu, alpha=1-decay)

    @torch.no_grad()
    def copy_to(self, model):
        """
        Replace model parameters with the shadow parameters.
        Use `store` to save the original parameters and `restore` to restore them.
        """
        self.to(self.device)
        self.del_redundant_params(model)
        model_params  = dict(model.named_parameters())
        shadow_params = dict( self.named_buffers()   )
        for key in model_params:
            if not model_params[key].requires_grad:
                assert key not in self.sname_lookup
                continue
            
            if key in self.sname_lookup:
                shadow_name = self.sname_lookup[key]
                model_params[key].data.copy_(shadow_params[shadow_name].to(model_params[key]))
            else:
                print(f'Warning: {key} not in shadow params')

    @torch.no_grad()
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.to(self.device)
        self.collected_params = [self.detach_to(p) for p in parameters]

    @torch.no_grad()
    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        self.to(self.device)
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.to(param))

    @torch.no_grad()
    def swap_state(self, model):
        """
        Swap model parameters and shadow parameters.
        This is RAM efficient compared to `copy_to` and `restore`
          since no new tensors are created.
        """
        self.to(self.device)
        self.del_redundant_params(model)
        
        # single_gpu = not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1 # this isn't working as expected
        is_rank_zero = self.rank == 0
        
        # swap rank 0 params with shadow params
        if is_rank_zero:
            self._swap_state_local(model)
        
        # using try catch since single_gpu is not working as expected
        # probably because of the way pytorch lightning handles distributed
        try:
            # wait for rank 0 to finish swapping
            torch.distributed.barrier()
        except RuntimeError as e:
            #traceback.print_exc()
            pass
        else:
            # send rank 0 params to other ranks
            self.sync_parameters(model)
    
    @torch.no_grad()
    def _swap_state_local(self, model):
        self.to(self.device)
        self.del_redundant_params(model)
        model_params  = dict(model.named_parameters())
        shadow_params = dict( self.named_buffers()   )
        for key in model_params:
            if not model_params[key].requires_grad:
                assert key not in self.sname_lookup
                continue
            shadow_name = self.sname_lookup[key]
            
            # get clone of model param
            model_param_clone = model_params[key].data.clone()
            
            # replace model param with shadow param
            model_params[key].data.copy_(
                shadow_params[shadow_name].data
                .to(model_params[key]))
            
            # replace shadow param with model param clone
            shadow_params[shadow_name].data.copy_(
                model_param_clone.data
                .to(shadow_params[shadow_name]))
            
            # delete model param clone
            del model_param_clone
    
    @torch.no_grad()
    def sync_parameters(self, model, src_rank=0):
        """
        Syncronize weights of model across GPUs.
        All GPUs with rank > 0 will copy the weights from rank 0.
        """
        model_params  = dict(model.named_parameters())
        for key in model_params:
            if not model_params[key].requires_grad:
                assert key not in self.sname_lookup
                continue
            
            param = model_params[key]
            torch.distributed.broadcast(param, src_rank)