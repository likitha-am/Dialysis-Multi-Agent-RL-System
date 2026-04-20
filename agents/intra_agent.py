import torch
import torch.nn as nn

from encoders.numeric_encoder import NumericEncoder
from encoders.categorical_encoder import CategoricalEncoder
from encoders.image_encoder import ImageEncoder
from encoders.text_encoder import TextEncoder
from encoders.fusion import FusionLayer

from communication.message_decoder import MessageDecoder
from models.agent_core import AgentCore


class IntraDialysisAgent(nn.Module):
    def __init__(self):
        super(IntraDialysisAgent, self).__init__()

        self.num_encoder = NumericEncoder(input_dim=5, output_dim=32)
        self.cat_encoder = CategoricalEncoder(num_categories=5, embed_dim=16)
        self.img_encoder = ImageEncoder(output_dim=64)
        self.txt_encoder = TextEncoder(vocab_size=1000, embed_dim=32, hidden_dim=64)

        fusion_input_dim = 32 + 16 + 64 + 64
        self.fusion = FusionLayer(input_dim=fusion_input_dim, output_dim=128)

        self.msg_decoder = MessageDecoder(message_dim=4, output_dim=128)

        self.core = AgentCore(action_dim=4)

    def forward(self, x_num, x_cat, x_img, x_txt, message):

        h_num = self.num_encoder(x_num)
        h_cat = self.cat_encoder(x_cat)
        h_img = self.img_encoder(x_img)
        h_txt = self.txt_encoder(x_txt)

        h = self.fusion(h_num, h_cat, h_img, h_txt)

        decoded_msg = self.msg_decoder(message)

        action, new_message = self.core(h, decoded_msg)

        return action, new_message