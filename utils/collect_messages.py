import torch

from agents.pre_agent import PreDialysisAgent
from agents.intra_agent import IntraDialysisAgent
from agents.post_agent import PostDialysisAgent
from environment.dialysis_env import DialysisEnv


def collect():

    pre_agent = PreDialysisAgent()
    intra_agent = IntraDialysisAgent()
    post_agent = PostDialysisAgent()

    batch_size = 8
    env = DialysisEnv(batch_size=batch_size)

    all_messages = []
    all_labels = []
    all_rewards = []

    for episode in range(20):

        state = env.reset()

        for step in range(5):

            noise = lambda x: torch.randn_like(x) * 0.1

            x_pre_num = torch.stack([
                state["bp"], state["fluid"],
                noise(state["bp"]), noise(state["bp"]),
                noise(state["bp"]), noise(state["bp"]),
                noise(state["bp"])
            ], dim=1)

            x_pre_cat = torch.zeros(batch_size, 3, dtype=torch.long)
            x_pre_img = torch.randn(batch_size, 3, 224, 224)
            x_pre_txt = torch.randint(0, 1000, (batch_size, 10))

            x_intra_num = torch.stack([
                state["bp"], state["fluid"],
                noise(state["bp"]), noise(state["bp"]),
                noise(state["bp"])
            ], dim=1)

            x_intra_cat = torch.zeros(batch_size, 1, dtype=torch.long)
            x_intra_img = torch.randn(batch_size, 3, 224, 224)
            x_intra_txt = torch.randint(0, 1000, (batch_size, 10))

            x_post_num = torch.stack([
                state["bp"], state["fluid"],
                noise(state["bp"]), noise(state["bp"])
            ], dim=1)

            x_post_cat = torch.zeros(batch_size, 1, dtype=torch.long)
            x_post_img = torch.randn(batch_size, 3, 224, 224)
            x_post_txt = torch.randint(0, 1000, (batch_size, 10))

            m0 = torch.zeros(batch_size, 4)

            # Forward
            a_pre, m1 = pre_agent(x_pre_num, x_pre_cat, x_pre_img, x_pre_txt, m0)
            a_intra, m2 = intra_agent(x_intra_num, x_intra_cat, x_intra_img, x_intra_txt, m1)
            a_post, m3 = post_agent(x_post_num, x_post_cat, x_post_img, x_post_txt, m2)

            # Store messages + labels
            all_messages.append(m1.detach())
            all_labels += ["Pre"] * batch_size

            all_messages.append(m2.detach())
            all_labels += ["Intra"] * batch_size

            all_messages.append(m3.detach())
            all_labels += ["Post"] * batch_size

            # Step env
            state, reward = env.step(a_pre, a_intra, a_post)

            # Store reward (repeat for each agent)
            all_rewards += [reward.item()] * (3 * batch_size)

            state = {
                "bp": state["bp"].detach(),
                "fluid": state["fluid"].detach()
            }

    messages = torch.cat(all_messages, dim=0)

    torch.save(messages, "messages.pt")
    torch.save(all_labels, "labels.pt")
    torch.save(all_rewards, "rewards.pt")

    print("Saved everything:", messages.shape)


if __name__ == "__main__":
    collect()