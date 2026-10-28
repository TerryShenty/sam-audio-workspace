"""Schedule validation from committed global sampler progress, without rounding drift."""
class HalfEpoch:
    def __init__(self, sample_count, cursor):
        assert sample_count>0 and cursor>=0
        self.n=sample_count
        self.next=2*cursor//sample_count+1
        previous=self.next-1
        # A checkpoint saved immediately before a boundary validation must not skip it.
        if previous and 0<=cursor-(previous*sample_count+1)//2<8:self.next=previous
    def due(self,cursor):return 2*cursor>=self.next*self.n
    def complete(self,cursor):
        assert self.due(cursor)
        self.next=2*cursor//self.n+1
