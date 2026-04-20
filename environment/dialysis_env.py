import torch

class DialysisEnv:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.bp = torch.randn(self.batch_size) * 10 + 120
        self.fluid = torch.rand(self.batch_size) * 5
        return self._get_state()

    def _get_state(self):
        return {
            "bp": self.bp,
            "fluid": self.fluid
        }

    def step(self, a_pre, a_intra, a_post):
        # Extract actions
        uf_pre = a_pre[:, 0]
        uf_intra = a_intra[:, 0]
        adjust_post = a_post[:, 0]

        # Update fluid
        self.fluid = self.fluid - (uf_pre * 0.3 + uf_intra * 0.7)

        # BP change
        self.bp = self.bp - (uf_pre + uf_intra) * 2
        self.bp = self.bp + adjust_post * 1.5

        # Clamp values
        self.bp = torch.clamp(self.bp, 60, 180)
        self.fluid = torch.clamp(self.fluid, 0, 10)

        # Reward
        bp_penalty = (self.bp - 120) ** 2
        fluid_penalty = (self.fluid) ** 2

        reward = - (bp_penalty + fluid_penalty).mean()

        return self._get_state(), reward