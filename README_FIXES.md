# Fix Updates for PyTorch & DGL Compatibility

This document details the changes made to the environment and codebase to fix dependency issues and allow legacy checkpoints to run on modern PyTorch versions.

## 1. Environment Updates
The following changes were made to the conda environment to resolve `ModuleNotFoundError`, `FileNotFoundError`, and version conflicts.

### Downgraded PyTorch and TorchData
The project was originally set up for older PyTorch versions. We standardized on **PyTorch 2.4.0** (CU121) to ensure compatibility with modern DGL and PyG libraries while maintaining stability.

```bash
# Uninstall conflicting versions
pip uninstall torch torchdata dgl -y

# Install PyTorch 2.4.0 with CUDA 12.1 support
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install compatible TorchData (downgraded to fix 'datapipes' error)
pip install torchdata==0.7.1
```

### Installed Compatible DGL (Deep Graph Library)
DGL is sensitive to PyTorch versions. We installed the specific pre-built wheel for PyTorch 2.4 / CUDA 12.1.
```bash
pip install dgl==2.4.0+cu121 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

### Installed PyTorch Geometric Dependencies
Required for the GNN components (`RGCNConv`, `TransformerConv`).
```bash
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

---

## 2. Code Patches in `eval.py`
To support loading older model checkpoints (likely from PyTorch 1.x) in this new environment, runtime patching was implemented in `eval.py`.

### Fix 1: Missing Inspector Module
Old checkpoints referenced `torch_geometric.nn.conv.utils.inspector`, which was removed in PyG 2.x.
*   **Action**: Monkeypatched `sys.modules` to include a dummy `Inspector` class so `pickle` can successfully deserialize the model.

### Fix 2: Validation of `_lazy_load_hook`
Old PyTorch `Linear` layers expected a `_lazy_load_hook` method that no longer exists on `torch.nn.Module`.
*   **Action**: Added a dummy `_lazy_load_hook` to `torch.nn.Module` to prevent `AttributeError` during loading.

### Fix 3: Runtime Model Migration (PyG 1.x -> 2.x)
Loaded models were missing internal attributes required by the new PyG `MessagePassing` engine. We implemented a post-load loop to:
1.  **Set `_decomposed_layers = 1`**.
2.  **Re-initialize `Inspector`**: Replaced the legacy/pickled `inspector` with a fresh `torch_geometric.inspector.Inspector` instance.
3.  **Regenerate `_user_args`**: Used the new `Inspector` to analyze the `message`, `aggregate`, and `update` methods and populate `_user_args`.
4.  **Fix Argument Collisions**: Specifically excluded `inputs` from the inspector's signature cache to prevent a `TypeError` in `aggregate()` where both positional and keyword arguments collided.

---

## 3. Running the Evaluation
With these fixes, the evaluation script can be run using the standard command:

```bash
python eval.py --dataset="iemocap_4" --modalities="atv"
```

### Expected Output
The script should load the model, process the dataset, and output an F1 score (approx **0.8320**).

```text
test: 100%|████████████| 1/1 [00:01<00:00,  1.58s/it]
...
F1 Score: 0.8320359557459978
```
