"""Only this entry point enables the bounded public demo policy."""
import torch
from app import create_app

torch.set_num_threads(2)
app = create_app(public_demo=True)
