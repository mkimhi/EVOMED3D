import torch

from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class TemporalModWarmupUnfreezeHook(Hook):
    """Switch a MoD adapter warmup run to joint DDP training.

    DDP only registers parameters that require gradients when it is created.
    The warmup therefore must rebuild the wrapper after unfreezing the
    pretrained detector; merely changing ``requires_grad`` would leave those
    parameters out of distributed gradient reduction.
    """

    def __init__(
        self,
        warmup_iters,
        find_unused_parameters=False,
        pretrained_lr_mult=1.0,
    ):
        self.warmup_iters = int(warmup_iters)
        self.find_unused_parameters = bool(find_unused_parameters)
        self.pretrained_lr_mult = float(pretrained_lr_mult)
        if not 0.0 < self.pretrained_lr_mult <= 1.0:
            raise ValueError('pretrained_lr_mult must be in (0, 1]')
        self._transitioned = False

    def _transition(self, runner, iteration):
        wrapped_model = runner.model
        model = wrapped_model.module if hasattr(wrapped_model, 'module') else wrapped_model
        unfreeze = getattr(model, 'unfreeze_temporal_mod_warmup', None)
        if unfreeze is None:
            raise AttributeError(
                'Temporal MoD adapter warmup requires '
                'model.unfreeze_temporal_mod_warmup')
        unfreeze()
        self._scale_pretrained_learning_rates(runner, model)

        # The optimizer was constructed before DDP and already owns all model
        # parameters.  Its frozen parameters have no state yet, so retaining
        # the optimizer preserves warmup momentum while adding state lazily for
        # the newly trainable pretrained weights.  Fixed detector constants
        # remain frozen and are therefore excluded from the rebuilt DDP graph.
        runner.optimizer.zero_grad()
        if hasattr(wrapped_model, 'module'):
            runner.model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=self.find_unused_parameters)

        self._transitioned = True
        runner.logger.info(
            'Temporal MoD adapter warmup finished at iteration %d; unfroze '
            'pretrained detector, set its LR multiplier to %.3g, and rebuilt '
            'DDP for joint training.', iteration, self.pretrained_lr_mult)

    def _scale_pretrained_learning_rates(self, runner, model):
        """Lower the LR of the weights released from the frozen warmup.

        MMCV stores the scheduler base LR separately from optimizer param
        groups, so both must be changed.  The MoD routers/adapters are absent
        from this recorded pretrained-name set and retain their warmup LR.
        """
        if self.pretrained_lr_mult == 1.0:
            return
        names = getattr(model, '_tm_warmup_unfreeze_parameter_names', ())
        pretrained_ids = {
            id(parameter) for name, parameter in model.named_parameters()
            if name in names
        }
        changed_indices = []
        for index, group in enumerate(runner.optimizer.param_groups):
            parameters = group.get('params', ())
            if parameters and all(id(parameter) in pretrained_ids for parameter in parameters):
                group['lr'] *= self.pretrained_lr_mult
                group['initial_lr'] = (
                    group.get('initial_lr', group['lr'] / self.pretrained_lr_mult)
                    * self.pretrained_lr_mult)
                changed_indices.append(index)

        for hook in runner.hooks:
            base_lr = getattr(hook, 'base_lr', None)
            if isinstance(base_lr, list):
                for index in changed_indices:
                    base_lr[index] *= self.pretrained_lr_mult

    def before_run(self, runner):
        # A resumed checkpoint is loaded before hooks run.  If it was created
        # after warmup, restore joint trainability and its reduced pretrained
        # LR before the first resumed forward pass.
        if not self._transitioned and runner.iter >= self.warmup_iters:
            self._transition(runner, runner.iter)

    def after_train_iter(self, runner):
        if self._transitioned or runner.iter + 1 < self.warmup_iters:
            return
        self._transition(runner, runner.iter + 1)
