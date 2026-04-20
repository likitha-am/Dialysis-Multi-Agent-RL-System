import torch
import torch.optim as optim

from agents.pre_agent import PreDialysisAgent
from agents.intra_agent import IntraDialysisAgent
from agents.post_agent import PostDialysisAgent

from environment.dialysis_env import DialysisEnv


def train():

    pre_agent = PreDialysisAgent()
    intra_agent = IntraDialysisAgent()
    post_agent = PostDialysisAgent()

    optimizer = optim.Adam(
        list(pre_agent.parameters()) +
        list(intra_agent.parameters()) +
        list(post_agent.parameters()),
        lr=1e-4
    )

    batch_size = 8
    env = DialysisEnv(batch_size=batch_size)

    for epoch in range(100):

        total_loss = 0

        # 🔥 multiple patients per epoch (variance reduction)
        for episode in range(5):

            state = env.reset()

            for step in range(5):

                noise = lambda x: torch.randn_like(x) * 0.1

                # -------- Pre --------
                x_pre_num = torch.stack([
                    state["bp"],
                    state["fluid"],
                    noise(state["bp"]),
                    noise(state["bp"]),
                    noise(state["bp"]),
                    noise(state["bp"]),
                    noise(state["bp"])
                ], dim=1)

                x_pre_cat = torch.zeros(batch_size, 3, dtype=torch.long)
                x_pre_img = torch.randn(batch_size, 3, 224, 224)
                x_pre_txt = torch.randint(0, 1000, (batch_size, 10))

                # -------- Intra --------
                x_intra_num = torch.stack([
                    state["bp"],
                    state["fluid"],
                    noise(state["bp"]),
                    noise(state["bp"]),
                    noise(state["bp"])
                ], dim=1)

                x_intra_cat = torch.zeros(batch_size, 1, dtype=torch.long)
                x_intra_img = torch.randn(batch_size, 3, 224, 224)
                x_intra_txt = torch.randint(0, 1000, (batch_size, 10))

                # -------- Post --------
                x_post_num = torch.stack([
                    state["bp"],
                    state["fluid"],
                    noise(state["bp"]),
                    noise(state["bp"])
                ], dim=1)

                x_post_cat = torch.zeros(batch_size, 1, dtype=torch.long)
                x_post_img = torch.randn(batch_size, 3, 224, 224)
                x_post_txt = torch.randint(0, 1000, (batch_size, 10))

                m0 = torch.zeros(batch_size, 4)

                # -------- Forward --------
                a_pre, m1 = pre_agent(x_pre_num, x_pre_cat, x_pre_img, x_pre_txt, m0)

                a_intra, m2 = intra_agent(
                    x_intra_num, x_intra_cat, x_intra_img, x_intra_txt, m1
                )

                a_post, m3 = post_agent(
                    x_post_num, x_post_cat, x_post_img, x_post_txt, m2
                )

                # -------- Environment --------
                state, reward = env.step(a_pre, a_intra, a_post)

                loss = -reward
                total_loss += loss

                # 🔥 cut graph each step
                state = {
                    "bp": state["bp"].detach(),
                    "fluid": state["fluid"].detach()
                }

        # 🔥 average loss over all episodes + steps
        loss = total_loss / (5 * 5)

        # -------- Backprop --------
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(pre_agent.parameters()) +
            list(intra_agent.parameters()) +
            list(post_agent.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()