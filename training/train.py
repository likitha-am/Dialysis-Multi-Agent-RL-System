import torch
import torch.optim as optim

from agents.pre_agent import PreDialysisAgent
from agents.intra_agent import IntraDialysisAgent
from agents.post_agent import PostDialysisAgent

from training.reward import compute_reward


def train():

    # Initialize agents
    pre_agent = PreDialysisAgent()
    intra_agent = IntraDialysisAgent()
    post_agent = PostDialysisAgent()

    # Optimizer (ALL params together)
    optimizer = optim.Adam(
        list(pre_agent.parameters()) +
        list(intra_agent.parameters()) +
        list(post_agent.parameters()),
        lr=1e-3
    )

    batch_size = 8

    for epoch in range(100):

        # ----------------------------
        # Dummy Data (replace later)
        # ----------------------------

        x_pre_num = torch.randn(batch_size, 7)
        x_pre_cat = torch.randint(0, 3, (batch_size, 3))
        x_pre_img = torch.randn(batch_size, 3, 224, 224)
        x_pre_txt = torch.randint(0, 1000, (batch_size, 10))

        x_intra_num = torch.randn(batch_size, 5)
        x_intra_cat = torch.randint(0, 3, (batch_size, 1))
        x_intra_img = torch.randn(batch_size, 3, 224, 224)
        x_intra_txt = torch.randint(0, 1000, (batch_size, 10))

        x_post_num = torch.randn(batch_size, 4)
        x_post_cat = torch.randint(0, 3, (batch_size, 1))
        x_post_img = torch.randn(batch_size, 3, 224, 224)
        x_post_txt = torch.randint(0, 1000, (batch_size, 10))

        # Initial message
        m0 = torch.zeros(batch_size, 4)

        # ----------------------------
        # Forward Pass
        # ----------------------------

        a_pre, m1 = pre_agent(x_pre_num, x_pre_cat, x_pre_img, x_pre_txt, m0)

        a_intra, m2 = intra_agent(
            x_intra_num, x_intra_cat, x_intra_img, x_intra_txt, m1
        )

        a_post, m3 = post_agent(
            x_post_num, x_post_cat, x_post_img, x_post_txt, m2
        )

        # ----------------------------
        # Compute Reward → Loss
        # ----------------------------

        reward = compute_reward(a_pre, a_intra, a_post)

        loss = -reward  # maximize reward

        # ----------------------------
        # Backprop
        # ----------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()