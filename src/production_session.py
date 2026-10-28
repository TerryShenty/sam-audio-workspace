"""Bind complete trainer operations to the accepted runtime lifecycle."""
class ProductionSession:
    def __init__(self,trainer,runtime,batch):
        assert runtime.activated
        self.trainer=trainer;self.runtime=runtime;self.batch=batch
        self._step=runtime.guard_step(lambda:trainer.step(batch))

    def step(self):
        return self._step()

    def checkpoint(self,directory):
        self.runtime.registry.require_step()
        self.runtime.registry.boundary()
        self.trainer.checkpoint(directory)

    def restore(self,manifest):
        def local_restore():
            # Read/validate once before invalidating state; writes are fail-closed.
            prepared=[]
            def validate():
                prepared.append(self.trainer.prepare_restore(manifest))
            def write():
                self.trainer.apply_restore(prepared.pop())
            self.runtime.lifecycle.restore(validate,write)
            self.runtime.registry.require_step()
        self.trainer.serial_restore(local_restore)
