# Data Preparation
## DAIR-V2X Dataset
DAIR-V2X is the first large-scale and real-world vehicle-infrastructure cooperative 3D object detection dataset. This dataset includes the DAIR-V2X-C, which has the cooperative view.
We train and evaluate the models on DAIR-V2X dataset. For downloading DAIR-V2X dataset, please refer to the guidelines in [DAIR-V2X](https://thudair.baai.ac.cn/cooptest). 

## Data Structure
### flow_data_jsons/*
We construct the frame pairs to generate the json files for FFNet training and evaluation. <br>
- flow_data_info_train_2.json: frame pairs constructing from DAIR-V2X-C to simulate the different latency for training, including k=1,2
- flow_data_info_val_n.json: frame pairs constructing from DAIR-V2X-C to simulate the different latency for evaluation. n=1,2,3,4,5, corresponding to the 100ms, 200ms, 300, 400ms and 500ms latency, respectively.
- example_flow_data_info_train_2.json: frame pairs constructing from the example dataset to simulate the different latency for training, including k=1,2
- example_flow_data_info_val_n.json: frame pairs constructing from the example dataset to simulate the different latency for evaluation. n=1,2,3,4,5, corresponding to the 100ms to 500ms latency.

### split_datas
The json files are used for spliting the dataset into train/val/test parts. <br>
Please refer to the [split_data](https://github.com/AIR-THU/DAIR-V2X/tree/main/data/split_datas) for the latest updates.

## Data Preprocess
We use the DAIR-V2X-C-Example to illustrate how we preprocess the dataset for our experiment. For the convenience of overseas users, we provide the original DAIR-V2X-Example dataset [here](https://drive.google.com/file/d/1bFwWGXa6rMDimKeu7yJazAkGYO8s4RSI/view?usp=sharing). We provide the preprocessed DAIR-V2X-C-Example dataset [here](https://drive.google.com/file/d/1y8bGwI63TEBkDEh2JU_gdV7uidthSnoe/view?usp=sharing). 

```
# Preprocess the dair-v2x-c dataset
python ./data/dair-v2x/preprocess.py --source-root ./data/dair-v2x/DAIR-V2X-Examples/cooperative-vehicle-infrastructure
```

##  Generate the Frame Pairs
We have provided the frame pair files in [flow_data_jsons](./flow_data_jsons). 
You can generate your frame pairs with the provided [example script](./frame_pair_generation.py).
```
# Preprocess the dair-v2x-c dataset
python ./data/dair-v2x/frame_pair_generation.py --source-root ./data/dair-v2x/DAIR-V2X-Examples/cooperative-vehicle-infrastructure
```