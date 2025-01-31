# Graph Generative Model


## File Overview
- **`utils.py`**: Contains utility functions (same as in the baseline).
- **`extract_feats.py`**: Extracts features from the text descriptions (same as in the baseline).
- **`autoencoder.py`**: Variational Graph Autoencoder (VGAE) implementation (inspired by the baseline).
- **`normalizing_flow.py`**: Implementation of normalizing flow models for graph generation. This code is designed to easily replace the DDPM model in the baseline by modifying the initialization, loss, and sampling lines in the original `main.py` file.
- **`da_utils.py`**: Contains utility functions for data analysis and visualization.
- **`GAN` folder**: Contains the GAN approach implementation.

## Notebooks and Usage
- **`main.ipynb`**: Implements training and testing of the proposed VGAE model.
- **`da.ipynb`**: Perform exploratory data analysis. 

### Notes
**Normalizing Flow and GAN Models**: These approaches were explored but did not yield conclusive results.
# Graph_Generation
