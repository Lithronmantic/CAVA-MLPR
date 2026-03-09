import torch.nn as nn
from .temporal_encoder import SimpleTemporalEncoder
from .offline_encoder import OfflineEncoder
from ..fusion.ib_fusion import InformationBottleneckFusion
from ..fusion.cfa_fusion import CFAFusion
from ..mil.dynamic_routing import DynamicRoutingMIL

class AVTopDetector(nn.Module):
    def __init__(self, d_model: int = 128, num_classes: int = 2,
                 fusion_type: str = "ib", ib_beta: float = 0.0,
                 use_offline: bool = False, d_v_in: int = None, d_a_in: int = None,
                 mil_tau: float = 2.0, mil_topk: int = None):
        super().__init__()
        if use_offline:
            assert d_v_in and d_a_in, "offline encoder needs d_v_in & d_a_in"
            self.enc = OfflineEncoder(d_v_in, d_a_in, d_model=d_model)
        else:
            self.enc = SimpleTemporalEncoder(d_model=d_model)

        if fusion_type == "ib":
            self.fusion = InformationBottleneckFusion(d_in=2*d_model, d_latent=d_model, beta=ib_beta)
            self.is_ib = True
        elif fusion_type == "cfa":
            self.fusion = CFAFusion(d_in=d_model)
            self.is_ib = False
        else:
            raise ValueError(f"unknown fusion type: {fusion_type}")

        self.mil = DynamicRoutingMIL(d_in=d_model, num_classes=num_classes, tau=mil_tau, topk=mil_topk)

    def set_ib_beta(self, beta: float):
        if self.is_ib and hasattr(self.fusion, "set_ib_beta"):
            self.fusion.set_ib_beta(beta)

    def forward(self, audio, video):
        v, a = self.enc(video, audio)
        z, aux = self.fusion(v, a) if not self.is_ib else (self.fusion(v, a)["z"], self.fusion(v, a))
        mil_out = self.mil(z)
        mil_out["z"] = z
        mil_out["clip_features"] = (z * mil_out["weights"].unsqueeze(-1)).sum(dim=1)
        if self.is_ib:
            mil_out.setdefault("ib_kl", aux.get("kl", v.new_tensor(0.0)))
            mil_out.setdefault("ib_recon", aux.get("recon", v.new_tensor(0.0)))
        else:
            mil_out.update(aux)  # gv/ga
        return mil_out
