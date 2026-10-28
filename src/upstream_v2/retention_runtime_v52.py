"""Count successful Qwen forwards separately from B2 input-capture pre-hooks.

condition_batch invokes the model pre-hooks twice to capture prepared inputs,
raising _PreparedInputCaptured before either model body runs. Its third call
executes the batched language forward. Selection itself is unchanged from v51.
Only scalars are retained by these hooks, never tensors or autograd graphs.
"""
from retention_scope_v51 import allows_ret25

FIELDS = ('calls', 'retained_layer_selections', 'checkpoint_layer_selections', 'fallback_calls')

class Retention:
 def __init__(self, model):
  assert len(model.layers) == 48
  self.count = 0
  self.attempts = dict.fromkeys(FIELDS, 0)
  self.completed = dict.fromkeys(FIELDS, 0)
  self.selection = None
  self.handle = model.register_forward_pre_hook(self.select, with_kwargs=True)
  # Default always_call=False: interrupted input captures cannot count as work.
  self.completed_handle = model.register_forward_hook(self.complete, with_kwargs=True)

 def configure(self, cfg):
  n = cfg['retained']; assert type(n) is int and 0 <= n <= 25
  if n == 25: assert allows_ret25(cfg), 'RET25_OUTSIDE_BOUNDED_SCOPE'
  self.count = n

 def select(self, model, args, kwargs):
  x = kwargs.get('inputs_embeds')
  if x is None: x = kwargs.get('input_ids')
  if x is None and args: x = args[0]
  assert x is not None
  b, l = x.shape[:2]; eligible = l <= 1024 and b*l <= 2048
  n = self.count if eligible else 0
  for i, layer in enumerate(model.layers): layer.gradient_checkpointing = i >= n
  self.selection = (n, int(not eligible))
  self._increment(self.attempts, *self.selection)

 @staticmethod
 def _increment(counters, retained, fallback):
  counters['calls'] += 1
  counters['retained_layer_selections'] += retained
  counters['checkpoint_layer_selections'] += 48-retained
  counters['fallback_calls'] += fallback

 def complete(self, model, args, kwargs, output):
  assert self.selection is not None and output is not None, 'RETENTION_COMPLETION_WITHOUT_SELECTION'
  self._increment(self.completed, *self.selection)
  self.selection = None

 def statistics(self):
  return dict(self.attempts, requested=self.count, completed=dict(self.completed))

 def record_window(self, row, before, cfg, *, require_execution=True):
  after = self.statistics()
  attempts = {k:after[k]-before[k] for k in FIELDS}
  completed = {k:after['completed'][k]-before['completed'][k] for k in FIELDS}
  for delta in (attempts, completed):
   assert all(v >= 0 for v in delta.values()), 'RETENTION_COUNTER_RESET'
   assert delta['retained_layer_selections']+delta['checkpoint_layer_selections'] == 48*delta['calls']
  assert all(completed[k] <= attempts[k] for k in FIELDS), 'RETENTION_COMPLETION_EXCEEDS_SELECTION'
  if require_execution:
   assert completed['calls'] == row['steps'], 'RETENTION_MODEL_FORWARD_COUNT'
   assert attempts['calls'] == 3*row['steps'], 'RETENTION_B2_INPUT_CAPTURE_COUNT'
   assert completed['fallback_calls'] == 0 and completed['retained_layer_selections'] == cfg['retained']*row['steps'], 'RETENTION_REQUEST_NOT_EXECUTED'
   assert self.selection is None, 'RETENTION_INCOMPLETE_LAST_FORWARD'
  row.update(retention_selection=after, retention_calls=completed,
             retention_pre_hook_calls=attempts,
             retention_pre_hooks_without_completed_forward=attempts['calls']-completed['calls'],
             retention_counts_are_selections_not_memory_bytes=True)

 def remove(self):
  self.handle.remove(); self.completed_handle.remove(); self.selection = None
