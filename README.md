# Official implementation of IIRG (Item-Item Relation Generation)

----------

### 📄 Paper Information 📄

- **[Title]** On the Memorization Behavior of LLMs in Generative Recommendation: Observations, Implications, and Training Strategies
- **[Authors]** Sunwoo Kim, Sunkyung Lee, Clark Mingxuan Ju, Donald Loveland, Bhuvesh Kumar, Kijung Shin, Neil Shah, and Liam Collins.
- **[Affiliations]** Snap Inc., KAIST, and Sungkyunkwan University
- **[TL;DR]** We train an LLM to perform next-item prediction and two types of neighbor generation tasks: (1) collaborative neighbor generation and (2) semantic neighbor generation.

### 📁 Datasets 📁

- **[Overview]** We support three Amazon datasets: Sports, Toys, and Beauty.
- **[Download]** Necessary files are located in [DropboxLink](https://www.dropbox.com/scl/fo/4bhh8b2kqow0bnskkmocv/ACNqaBkWXztsWcgyZqbATb8?rlkey=yhld5vyrv3f7jkmmiqld05ans&st=7gqqqhco&dl=0).

### 💻 Repository Overview 💻

- **[ID Information]** Our implementation supports two popular types of item identifiers: semantic IDs (SIDs) and term IDs (TIDs).
- **[LLM Backbone]** We use the Qwen family as the backbone LLM for generative recommender systems.
- **[Code Structure]** The repository is organized as follows:
  - **[Step 1]** Build prerequisite files. Once generated, these files can be used for both identifier types.
  - **[Step 2]** Build item identifiers using either SIDs or TIDs.
  - **[Step 3]** Construct the training data.
  - **[Step 4]** Train LLM-based recommender systems.
  - **[Step 5]** Evaluate the trained model.

### ⚙️ Installation ⚙️
- **[Requirement installation]** You can install the required dependencies using the commands below:
```
conda create -n iirg_env python=3.12.11 -y
conda activate iirg_env
python -m pip install -r requirements.txt
```
- **[Note]** Each LLM training script launches a virtual environment named `iirg_env` at the beginning of the code. If you use a different environment name, please update that part accordingly.

### 📘 How to Use 📘
- **[Package versions]** Please refer to ```requirement.txt``` file.
- **[Prerequisite Step 1]** Locate (1) item meta information and (2) user-item sequential information. Related files can be downloaded from the [DropboxLink](https://www.dropbox.com/scl/fo/4bhh8b2kqow0bnskkmocv/ACNqaBkWXztsWcgyZqbATb8?rlkey=yhld5vyrv3f7jkmmiqld05ans&st=7gqqqhco&dl=0). Necessary files are as follows:
```
./data
  sports.item.json
  sports_sequential_data.txt
  toys.item.json
  toys_sequential_data.txt
  beauty.item.json
  beauty_sequential_data.txt
```
- **[Prerequisite Step 2]** Run ```python3 common_step1_embedding_builder.py --dataset <dataname>```.
  - **[Detail]** This step builds an embedding of each item from its title and description.
- **[Prerequisite Step 3]** Run ```python3 common_step2_neighbor_builder.py --dataset <dataname>```.
  - **[Detail]** This step builds (1) collaborative neighbors, (2) semantic neighbors, and (3) item popularity files.
- **[Next Step]** After generating neighbors, the next step depends on the type of item ID used. Please refer to the README file in either the `sid_based` or `tid_based` folder for the corresponding instructions.

### 🙏 Acknowledgements 🙏
- **[Codebase]** Our implementation builds upon the codebase originally developed in [GRLM](https://github.com/ZY0025/GRLM). We sincerely thank the authors for sharing their well-structured codebase.
- **[LLM Fine-tuning]** We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for LLM fine-tuning.

