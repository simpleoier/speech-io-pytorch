# speech-io-pytorch
A DataLoader in pytorch for speech recognition


## Usage:
```python
from dataset import Dataset
from dataLoader import DataLoader
import torch

feat_config_parms  = [dict(file_name=scp_file_name,
                           type="SCP",
                           format="HTK",
                           dim=input_dim,
                           context_window=(leftFeatContext, rightFeatContext))
                     ]
label_config_parms = [dict(file_name=json_label_name,
                           type="JSON",
                           dim=output_dim,
                           label_type="category")
                     ]

dataset = Dataset(feat_config_parms, label_config_parms, max_utt_len, verify_length=False)
dataLoaderIter = iter(DataLoader(dataset,
                                 batch_size=batch_size,
                                 random_size=random_size,
                                 epoch_size=epoch_size,
                                 random_seed=seed,
                                 frame_mode=frame_mode,
                                 padding_value=0))
                                 
for batch_cnt, (batch_data, batch_lengths, keys) in enumerate(dataLoaderIter):
    feature = batch_data[0][0]  # Tensor
    label   = batch_data[1][0]  # Tensor
    feature_lengths = batch_lengths[0][0]
    label_lengths   = batch_lengths[1][0]
```

