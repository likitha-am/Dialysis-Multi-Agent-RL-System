import torch
import torch.optim as optim

from agents.pre_agent import PreDialysisAgent
from agents.intra_agent import IntraDialysisAgent
from agents.post_agent import PostDialysisAgent

from environment.dialysis_env import DialysisEnv


def train():

    # Agents
    pre_agent = PreDialysisAgent()
    intra_agent = IntraDialysisAgent()
    post_agent = PostDialysisAgent()

    optimizer = optim.Adam(
        list(pre_agent.parameters()) +
        list(intra_agent.parameters()) +
        list(post_agent.parameters()),
        lr=1e-3
    )

    batch_size = 8

    # Environment
    env = DialysisEnv(batch_size=batch_size)
    state = env.reset()

    for epoch in range(100):

        # ----------------------------
        # Convert state → inputs
        # ----------------------------

        # Pre agent input (7 features)
        x_pre_num = torch.stack([
            state["bp"],
            state["fluid"],
            torch.zeros_like(state["bp"]),
            torch.zeros_like(state["bp"]),
            torch.zeros_like(state["bp"]),
            torch.zeros_like(state["bp"]),
            torch.zeros_like(state["bp"])
        ], dim=1)

        x_pre_cat = torch.zeros(batch_size, 3, dtype=torch.long)
        x_pre_img = torch.randn(batch_size, 3, 224, 224)
        x_pre_txt = torch.randint(0, 1000, (batch_size, 10))

        # Intra agent input (5 features)
        x_intra_num = torch.stack([
            state["bp"],
            state["fluid"],
            torch.zeros_like(state["bp"]),
            torch.zeros_like(state["bp"]),
            torch.zeros_like(state["bp"])
        ], dim=1)

        x_intra_cat = torch.zeros(batch_size, 1, dtype=torch.long)
        x_intra_img = torch.randn(batch_size, 3, 224, 224)
        x_intra_txt = torch.randint(0, 1000, (batch_size, 10))

        # Post agent input (4 features)
        x_post_num = torch.stack([
            state["bp"],
            state["fluid"],
            torch.zeros_like(state["bp"]),
            torch.zeros_like(state["bp"])
        ], dim=1)

        x_post_cat = torch.zeros(batch_size, 1, dtype=torch.long)
        x_post_img = torch.randn(batch_size, 3, 224, 224)
        x_post_txt = torch.randint(0, 1000, (batch_size, 10))

        # Initial message
        m0 = torch.zeros(batch_size, 4)

        # ----------------------------
        # Forward pass
        # ----------------------------

        a_pre, m1 = pre_agent(x_pre_num, x_pre_cat, x_pre_img, x_pre_txt, m0)

        a_intra, m2 = intra_agent(
            x_intra_num, x_intra_cat, x_intra_img, x_intra_txt, m1
        )

        a_post, m3 = post_agent(
            x_post_num, x_post_cat, x_post_img, x_post_txt, m2
        )

        # ----------------------------
        # Environment step (FIXED)
        # ----------------------------

        state, reward = env.step(
    a_pre.detach(),
    a_intra.detach(),
    a_post.detach()
)

        loss = -reward + 0.001 * (
    a_pre.pow(2).mean() +
    a_intra.pow(2).mean() +
    a_post.pow(2).mean()
)

        # ----------------------------
        # Backprop
        # ----------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()