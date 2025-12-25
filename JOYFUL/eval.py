import pickle
import argparse
import torch
from sklearn import metrics
from tqdm import tqdm
import joyful

log = joyful.utils.get_logger()


import sys
from types import ModuleType

# Monkeypatch for legacy torch_geometric checkpoints
try:
    import torch_geometric.nn.conv.utils.inspector
except ImportError:
    # Create a dummy module
    mock_inspector = ModuleType("torch_geometric.nn.conv.utils.inspector")
    
    # Define a dummy Inspector class which is likely what is being looked up
    class Inspector:
        def __init__(self, base_class):
            pass
        def inspect(self, *args, **kwargs):
            return {}
        def keys(self, *args, **kwargs):
            return []
        def implements(self, *args, **kwargs):
            return False
            
    mock_inspector.Inspector = Inspector
    
    # Inject into sys.modules
    sys.modules["torch_geometric.nn.conv.utils.inspector"] = mock_inspector

# Monkeypatch for Module._lazy_load_hook (legacy PyTorch checkpoint issue)
import torch.nn
def dummy_lazy_load_hook(self, *args, **kwargs):
    pass
setattr(torch.nn.Module, "_lazy_load_hook", dummy_lazy_load_hook)
print(f"DEBUG: Patched Module._lazy_load_hook: {hasattr(torch.nn.Module, '_lazy_load_hook')}")

def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def main(args):
    # data = load_pkl(f"./data/iemocap_4/data_iemocap_4.pkl")
    data = load_pkl(f"./data/iemocap/data_iemocap.pkl")
    # model_dict = torch.load('./model_checkpoints/iemocap_4_best_dev_f1_model_atv.pt')
    model_dict = torch.load('model_checkpoints/iemocap_best_dev_f1_model_atv.pt')
    stored_args = model_dict["args"]
    model = model_dict["modelN_state_dict"]
    
    # Post-load patch for legacy PyG models
    from torch_geometric.inspector import Inspector
    for m in model.modules():
        if not hasattr(m, "decomposed_layers"):
            # Set internal attribute directly to bypass property setter checks
            m._decomposed_layers = 1
        if not hasattr(m, "explain"):
            m.explain = False
        
        
        # Patch for missing inspector-generated attributes
        if hasattr(m, "message"): # Check for MessagePassing-like modules
            # Overwrite the legacy inspector with a new compatible one
            m.inspector = Inspector(m.__class__)
            
            # Legacy models might not have special_args, default to empty set
            special_args = getattr(m, 'special_args', set())
            
            # Exclusion for inspection: remove 'inputs' to avoid collision, but KEEP special_args (like index) 
            inspect_exclude = {'inputs'}
            
            # Manually inspect signatures with LIMITED exclusion
            m.inspector.inspect_signature(m.message, exclude=inspect_exclude)
            m.inspector.inspect_signature(m.aggregate, exclude=inspect_exclude)
            if hasattr(m, "message_and_aggregate"):
                m.inspector.inspect_signature(m.message_and_aggregate, exclude=inspect_exclude)
            m.inspector.inspect_signature(m.update, exclude=inspect_exclude)
            if hasattr(m, "edge_update"): 
               m.inspector.inspect_signature(m.edge_update, exclude=inspect_exclude)
            
            # Populate user_args using the inspector (filtering out special_args as usual)
            if not hasattr(m, "_user_args"):
                m._user_args = m.inspector.get_flat_param_names(
                    funcs=['message', 'aggregate', 'update'],
                    exclude=special_args
                )
            if not hasattr(m, "_fused_user_args"):
                m._fused_user_args = m.inspector.get_flat_param_names(
                    funcs=['message_and_aggregate', 'update'],
                    exclude=special_args
                )
            
    modelF = model_dict["modelF_state_dict"]
    testset = joyful.Dataset(data["test"], modelF, False, stored_args)
    test = True
    with torch.no_grad():
        golds = []
        preds = []
        for idx in tqdm(range(len(testset)), desc="test" if test else "dev"):
            data = testset[idx]
            golds.append(data["label_tensor"])
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(stored_args.device)
            y_hat = model(data, False)
            preds.append(y_hat.detach().to("cpu"))

        if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
            golds = torch.cat(golds, dim=0).numpy()
            preds = torch.cat(preds, dim=0).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
        else:
            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")

        if test:
            print(metrics.classification_report(golds, preds, digits=4))

            if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
                happy = metrics.f1_score(golds[:, 0], preds[:, 0], average="weighted")
                sad = metrics.f1_score(golds[:, 1], preds[:, 1], average="weighted")
                anger = metrics.f1_score(golds[:, 2], preds[:, 2], average="weighted")
                surprise = metrics.f1_score(
                    golds[:, 3], preds[:, 3], average="weighted"
                )
                disgust = metrics.f1_score(golds[:, 4], preds[:, 4], average="weighted")
                fear = metrics.f1_score(golds[:, 5], preds[:, 5], average="weighted")

                f1 = {
                    "happy": happy,
                    "sad": sad,
                    "anger": anger,
                    "surprise": surprise,
                    "disgust": disgust,
                    "fear": fear,
                }

            print(f"F1 Score: {f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval.py")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="iemocap_4",
        choices=["iemocap", "iemocap_4", "mosei", "meld"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="Computing device.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in seqcontext.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        # required=True,
        choices=["a", "at", "atv", "t", "v", "av"],
        help="Modalities",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    args = parser.parse_args()
    main(args)