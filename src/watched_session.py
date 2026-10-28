"""Attach worker progress protection without altering training or restore semantics."""
class WatchedSession:
    def __init__(self,session,watchdog,*,phase):
        assert phase in ('qualify','train')
        self.session=session;self.watchdog=watchdog;self.phase=phase
        self.trainer=session.trainer;self.runtime=session.runtime;self.batch=session.batch
        self.watchdog.tick(phase,self.trainer.step_number)
    def step(self):
        before=self.trainer.step_number
        self.watchdog.tick(self.phase,before)
        result=self.session.step()
        assert self.trainer.step_number==before+1,'UNCOMMITTED_STEP'
        self.watchdog.tick(self.phase,self.trainer.step_number)
        return result
    def checkpoint(self,directory):
        self.watchdog.tick('checkpoint',self.trainer.step_number)
        result=self.session.checkpoint(directory)
        self.watchdog.tick(self.phase,self.trainer.step_number)
        return result
    def restore(self,manifest):
        self.watchdog.tick('restore',self.trainer.step_number)
        result=self.session.restore(manifest)
        self.watchdog.tick(self.phase,self.trainer.step_number)
        return result
