# Official Implementation of TID-based IIRG

This branch provides an LLM-based generative recommender system that uses term IDs (TIDs) as item representations.

### 🚨 Key Notifications 🚨

- **[Notification 1]** Before running the SID-based pipeline, make sure to run `common_step1_embedding_builder.py` and `common_step2_neighbor_builder.py`.

- **[Notification 2]** We provide pre-built item TIDs. These files are available at this [Dropbox link](https://www.dropbox.com/scl/fo/jgiz3otjg4ym57x64iebp/AEWDocBpLXbBbTLT4mHAEoM?rlkey=3bkztsngzfjv7cm8p0jzxwydp&st=yzidqq50&dl=0).

  - **[Detail]** `{dataname}_id2meta_text.json` provides the pre-computed mapping between TIDs and items.
  - **[How to Use This Data]** Place these files in the `./tid_based/data` folder in this directory.
  - **[How to Train an LLM with This Data]** After placing these files in the data directory, you can skip **Steps 1–3** in the **How to Use** section below and start from **Step 4**.

### 📘 How to Use 📘

- **[Step 1. Build TIDs]** Run `python3 step1_tid_draft_builder.py --dataset <dataname>`.
  - **[Detail]** This generates our own TID, which prompts an LLM to generate the hierarchical keyword structure.

- **[Step 2. Perform TID reflection]** Run `python3 step2_tid_reflection.py --dataset <dataname>`.
  - **[Detail]** This refines generated TID draft to improve the consistency of terms across different items. This is performed based on LLMs.

- **[Step 3. Build the TID–Item Mapping]** Run `python3 step3_tid_mapper_builder.py --dataset <dataname>`.
  - **[Detail]** This generates the mapping between each item and its corresponding TID, which is used to build subsequent training data.

- **[Step 4. Build LLM Training and Evaluation Data]** Run the following three scripts:
  - **[Run 1]** `python3 step4_tid_next_item_prediction_task_builder.py --dataset <dataname>`
  - **[Run 2]** `python3 step4_tid_collaborative_neighbor_task_builder.py --dataset <dataname>`
  - **[Run 3]** `python3 step4_tid_semantic_neighbor_task_builder.py --dataset <dataname>`
  - **[Detail]** These scripts generate training and evaluation samples for the next-item prediction task, as well as training samples for the collaborative-neighbor generation and semantic-neighbor generation tasks.

- **[Step 5. Merge Data]** Run `python3 step5_merge_files.py --dataset <dataname>`.
  - **[Detail]** This merges samples from the three tasks into a single training dataset.
  - **🚨 [Key Notification] 🚨** You can control the relative weight of each task in the final loss function.

- **[Step 6. Train the LLM-based RecSys Model]** Run `bash step6_run_tid_llm.train.sh`.
  - **[Detail 1]** This trains the LLM-based generative recommender model using the constructed training dataset. The script is configured for two 80GB GPUs, so please adjust the dataset name and GPU-related settings according to your environment.
    - **[Tip]** For 40GB GPUs, we recommend processing 2 samples per GPU.
  - **[Detail 2]** In our experiments, we use a batch size of 128, with 4 samples per GPU per iteration, 16 gradient accumulation steps, and 2 GPUs.

- **[Step 7. Generate Recommendations]** Run `python3 step7_tid_evaluation_generation.py --dataset <dataname>`.
  - **[Detail]** This generates recommendations for each user. This step returns only the generated text outputs and does not directly map each recommendation to a specific item. Therefore, an additional decoding and metric calculation step is required.

- **[Step 8. Decode Outputs and Calculate Metrics]** Run `python3 step8_tid_metric_calculation.py --dataset <dataname>`.
  - **[Detail]** This decodes the generated outputs by parsing them and matching them to item IDs.