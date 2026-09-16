"""Optional terminal/notebook progress, independent of returned execution data."""


class CaseProgress:
    def __init__(self, total: int, *, enabled: bool, desc: str, unit: str = "case"):
        if type(enabled) is not bool:
            raise ValueError("show_progress must be a bool")
        self.total, self.enabled, self.desc, self.unit = total, enabled, desc, unit
        self.counts = {"succeeded": 0, "failed": 0}
        self.bar = None

    def __enter__(self):
        if self.enabled:
            from tqdm.auto import tqdm

            self.bar = tqdm(total=self.total, desc=self.desc, unit=self.unit, dynamic_ncols=True)
            self.bar.set_postfix(self.counts)
        return self

    def advance(self, status: str = "succeeded"):
        self.counts[status] = self.counts.get(status, 0) + 1
        if self.bar is not None:
            self.bar.set_postfix(self.counts, refresh=False)
            self.bar.update(1)

    def __exit__(self, *exc):
        if self.bar is not None:
            self.bar.close()
