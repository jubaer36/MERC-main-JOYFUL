
# AutoFusion adapted for correct behavior in Eval
import sys
from types import ModuleType
import pickle
import argparse
import torch
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm
import joyful
import os
import warnings
import numpy as np

# Monkeypatch for torch_geometric legacy support in checkpoints
try:
    import torch_geometric.nn.conv.utils.inspector
except ImportError:
    # Create dummy module structure
    inspector_module = ModuleType("torch_geometric.nn.conv.utils")
    inspector_module.inspector = ModuleType("torch_geometric.nn.conv.utils.inspector")
    
    class Inspector:
        def __init__(self, *args, **kwargs):
            pass
            
    inspector_module.inspector.Inspector = Inspector
    
    # We need to ensure the full path exists in sys.modules
    sys.modules["torch_geometric.nn.conv.utils.inspector"] = inspector_module.inspector

    # Also try simpler patch if the above structure is too deep/finicky
    # Some checkpoints look for torch_geometric.nn.conv.utils.inspector directly
    # Using a dummy class factory
    class MockInspector:
        def inspect(self, *args, **kwargs): return {}
        def implements(self, *args, **kwargs): return False
    
    sys.modules['torch_geometric.nn.conv.utils.inspector'] = ModuleType('inspector')
    sys.modules['torch_geometric.nn.conv.utils.inspector'].Inspector = MockInspector

# Monkeypatch for _lazy_load_hook in legacy PyTorch checkpoints
def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    pass
torch.nn.Module._lazy_load_hook = _lazy_load_hook

# Monkeypatch for _lazy_load_hook in legacy PyTorch checkpoints
def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    pass
torch.nn.Module._lazy_load_hook = _lazy_load_hook

def fix_legacy_modules(model):
    from torch_geometric.nn import RGCNConv, TransformerConv, GATConv, GCNConv
    # Helper to repair MessagePassing modules
    def repair_mp(module, dummy_cls, dummy_args):
        # Handle decomposed_layers property
        if not hasattr(module, '_decomposed_layers'):
            module._decomposed_layers = 1
        if not hasattr(module, 'decomposed_layers'):
            try: module.decomposed_layers = 1
            except: pass
        
        # Handle explain property
        if not hasattr(module, '_explain'):
            module._explain = False
            
        # Inspect dummy for MessagePassing internals
        try:
            dummy = dummy_cls(**dummy_args)
            for attr in ['_user_args', '_fuse_args', 'inspector', 'jittable']:
                if not hasattr(module, attr) and hasattr(dummy, attr):
                    setattr(module, attr, getattr(dummy, attr))
        except Exception as e:
            log.warning(f"Failed to create dummy {dummy_cls.__name__} for repair: {e}")

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if "RGCNConv" in cls_name:
             repair_mp(module, RGCNConv, {"in_channels":1, "out_channels":1, "num_relations":1})
             # Check for weight/root alias specific to RGCN
             if not hasattr(module, 'weight') and hasattr(module, 'root'):
                 module.weight = module.root
        
        elif "TransformerConv" in cls_name:
             # TransformerConv args: in_channels, out_channels
             repair_mp(module, TransformerConv, {"in_channels":1, "out_channels":1})
             
        elif "GCNConv" in cls_name: # Just in case
             repair_mp(module, GCNConv, {"in_channels":1, "out_channels":1})
             
        elif "GATConv" in cls_name: # Just in case
             repair_mp(module, GATConv, {"in_channels":1, "out_channels":1})

warnings.filterwarnings("ignore")
log = joyful.utils.get_logger()

class EvalAutoFusion(nn.Module):
    def __init__(self, a_dim=0, t_dim=0, v_dim=0):
        super(EvalAutoFusion, self).__init__()
        self.a_dim = a_dim
        self.t_dim = t_dim
        self.v_dim = v_dim
        
        input_features = a_dim + t_dim + v_dim
        self.input_features = input_features

        self.fuse_inGlobal = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.fuse_outGlobal = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, input_features)
        )
        
        num_active = 0
        if a_dim > 0: num_active += 1
        if t_dim > 0: num_active += 1
        if v_dim > 0: num_active += 1
        
        inter_input_features = 460 * num_active

        self.fuse_inInter = nn.Sequential(
            nn.Linear(inter_input_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.fuse_outInter = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, inter_input_features)
        )

        self.criterion = nn.MSELoss()

        if a_dim > 0:
            self.projectA = nn.Linear(a_dim, 460)
        if t_dim > 0:
            self.projectT = nn.Linear(t_dim, 460)
        if v_dim > 0:
            self.projectV = nn.Linear(v_dim, 460)
            
        self.projectB = nn.Sequential(
            nn.Linear(460, 460),
        )

    def forward(self, a=None, t=None, v=None):
        inputs = []
        if self.a_dim > 0:
            if a is None: raise ValueError("Expected audio input (a) but got None")
            inputs.append(a)
        
        if self.t_dim > 0:
            if t is None: raise ValueError("Expected text input (t) but got None")
            inputs.append(t)
            
        if self.v_dim > 0:
            if v is None: raise ValueError("Expected visual input (v) but got None")
            inputs.append(v)
            
        # Global
        cat_inputs = torch.cat(inputs, dim=-1)
        globalCompressed = self.fuse_inGlobal(cat_inputs)
        
        # Inter
        B = self.projectB(torch.ones(460).to(cat_inputs.device)) 
        inter_reps = []
        
        if self.a_dim > 0:
            A = self.projectA(a)
            BA = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), A), dim=1)
            bba = torch.mm(BA, torch.unsqueeze(A, dim=1)).squeeze(1)
            inter_reps.append(bba)

        if self.t_dim > 0:
            T = self.projectT(t)
            BT = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), T), dim=1)
            bbt = torch.mm(BT, torch.unsqueeze(T, dim=1)).squeeze(1)
            inter_reps.append(bbt)

        if self.v_dim > 0:
            V = self.projectV(v)
            BV = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), V), dim=1)
            bbv = torch.mm(BV, torch.unsqueeze(V, dim=1)).squeeze(1)
            inter_reps.append(bbv)

        cat_inter = torch.cat(inter_reps, dim=-1)
        interCompressed = self.fuse_inInter(cat_inter)
        
        return torch.cat((globalCompressed, interCompressed), 0), 0

def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def main(args):
    # Determine Data Path
    data_path = None
    if args.emotion:
        data_path = os.path.join(
            args.data_dir_path,
            args.dataset,
            "data_" + args.dataset + "_" + args.emotion + ".pkl",
        )
    else:
         data_path = os.path.join(
            args.data_dir_path, args.dataset, "data_" + args.dataset + ".pkl"
        )
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    log.info(f"Loading data from {data_path}")
    data = load_pkl(data_path)

    log.info(f"Loading checkpoint from {args.model_ckpt}")
    model_dict = torch.load(args.model_ckpt, map_location=args.device)
    
    # Checkpoint structure handling
    if "args" in model_dict:
        stored_args = model_dict["args"]
        stored_args.device = args.device # Override device
    else:
        # Fallback if no args (shouldn't happen for valid checkpoints)
        log.warning("No args in checkpoint, using current args")
        stored_args = args

    # Override dataset params if provided (critical for paths/dims)
    stored_args.dataset = args.dataset
    stored_args.modalities = args.modalities

    # Set dimensions map for manual fix (IEMOCAP override)
    input_dims_map = {
        "iemocap": {"a": 100, "t": 768, "v": 512},
        "iemocap_4": {"a": 100, "t": 768, "v": 512},
        "meld": {"a": 100, "t": 768, "v": 512},
        "mosei": {"a": 80, "t": 768, "v": 35}
    }
    cur_dims = input_dims_map.get(args.dataset, {"a":0, "t":0, "v":0})
    a_dim = cur_dims["a"] if "a" in args.modalities else 0
    t_dim = cur_dims["t"] if "t" in args.modalities else 0
    v_dim = cur_dims["v"] if "v" in args.modalities else 0

    # Handle modelF
    if "modelF_state_dict" in model_dict:
        modelF_raw = model_dict["modelF_state_dict"]
        # If it's a Module (full object), extract weights
        if not isinstance(modelF_raw, dict):
            state_dict = modelF_raw.state_dict()
        else:
            state_dict = modelF_raw
        
        # Initialize our robust fusion model
        modelF = EvalAutoFusion(a_dim, t_dim, v_dim)
        try:
            modelF.load_state_dict(state_dict)
            log.info("Loaded modelF weights into EvalAutoFusion")
        except Exception as e:
            log.warning(f"Strict load failed for modelF: {e}. Trying strict=False")
            modelF.load_state_dict(state_dict, strict=False)
    else:
        log.warning("No modelF found in checkpoint. Using random init EvalAutoFusion.")
        modelF = EvalAutoFusion(a_dim, t_dim, v_dim)

    # Handle main model
    if "modelN_state_dict" in model_dict:
        modelN_raw = model_dict["modelN_state_dict"]
        # If full object, use it directly (it's custom class JOYFUL)
        # Or preferably, init new and load weights to ensure code compatibility?
        # User snippet used `model = model_dict["modelN_state_dict"]` directly.
        # But if we modify args (dims), using the pickled object might retain old args.
        # Let's try using the pickled object first as user requested, 
        # BUT we must ensure it's on the right device.
        if not isinstance(modelN_raw, dict) and isinstance(modelN_raw, nn.Module):
            model = modelN_raw.to(args.device)
            log.info("Using pickled JOYFUL model object")
        else:
            # It's a dict, or we prefer to re-init
            # But the user snippet implies it's an object.
            # If it IS a dict, we must init JOYFUL.
            stored_args.dataset_embedding_dims = {
                 args.dataset: {
                    "a": 1024, "t": 1024, "v": 1024,
                    "at": 1024, "tv": 1024, "av": 1024, "atv": 1024,
                 }
            }
            # Hack: define these for fallback init
            if not hasattr(stored_args, 'concat_gin_gout'): stored_args.concat_gin_gout = False 
            
            model = joyful.JOYFUL(stored_args).to(args.device)
            if isinstance(modelN_raw, dict):
                model.load_state_dict(modelN_raw)
            else:
                model.load_state_dict(modelN_raw.state_dict())
            log.info("Initialized new JOYFUL model and loaded weights")
    else:
        raise ValueError("No modelN found in checkpoint")

    # Fix legacy issues in loaded modules (RGCNConv etc)
    fix_legacy_modules(model)
    log.info("Applied legacy module fixes")

    # Important: Setting eval mode
    model.eval()
    modelF.eval()

    testset = joyful.Dataset(data["test"], modelF, False, stored_args)
    test = True
    
    log.info("Starting evaluation loop...")
    with torch.no_grad():
        golds = []
        preds = []
        for idx in tqdm(range(len(testset)), desc="test"):
            data_batch = testset[idx]
            golds.append(data_batch["label_tensor"])
            for k, v in data_batch.items():
                if not k == "utterance_texts" and hasattr(v, 'to'):
                    data_batch[k] = v.to(args.device)
            
            # Forward pass
            # Note: Dataset.py alrady runs modelF during __getitem__ (padding) if configured so?
            # Wait, Dataset.__getitem__ calls padding -> raw_batch.
            # padding() calls modelF(a,t,v) to get embeddings. 
            # So modelF MUST be passed to Dataset.
            # We did: testset = joyful.Dataset(..., modelF, ...)
            
            y_hat = model(data_batch, False)
            preds.append(y_hat.detach().to("cpu"))

        if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
            golds = torch.cat(golds, dim=0).numpy()
            preds = torch.cat(preds, dim=0).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
        else:
            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")

        print(metrics.classification_report(golds, preds, digits=4))
        print(f"F1 Score: {f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval.py")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="iemocap_4", choices=["iemocap", "iemocap_4", "mosei", "meld"], help="Dataset name.")
    parser.add_argument("--data_dir_path", type=str, default="./data", help="Dataset directory path")
    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--modalities", type=str, default="atv", choices=["a", "at", "atv", "t", "v", "av", "tv"], help="Modalities")
    parser.add_argument("--emotion", type=str, default=None, help="emotion class for mosei")

    args = parser.parse_args()
    main(args)
