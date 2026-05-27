# Official Implementation of SID-based IIRG

This branch provides an LLM-based generative recommender system that uses semantic IDs (SIDs) as item representations.

### 🚨 Key Notifications 🚨

- **[Notification 1]** Before running the SID-based pipeline, make sure to run `common_step1_embedding_builder.py` and `common_step2_neighbor_builder.py`.

- **[Notification 2]** We provide pre-built item SIDs and pre-trained special token embeddings corresponding to SID tokens for the `Qwen-3.5-4B` model. These files are available at this [Dropbox link](https://www.dropbox.com/scl/fo/3otvejake74h0ronnmesx/AM8V_kFnYTmhCRXsxCYYjyY?rlkey=vm0sqp89clhbmk3ipptc9dye2&st=0vta4fa7&dl=0).

  - **[Detail 1]** `{dataname}_id2meta_codebook.json` provides the pre-computed mapping between SIDs and items.
  - **[Detail 2]** `{dataname}_special_tokens.json` provides the pre-computed codebooks across all items.
  - **[Detail 3]** `{dataname}_codebook_aligned_embeddings.safetensors` provides special token embeddings aligned with the `Qwen-3.5-4B` token space through continued pre-training, i.e., vocabulary expansion.
  - **[How to Use This Data]** Place these files in the `./sid_based/data` folder in this directory.
  - **[How to Train an LLM with This Data]** After placing these files in the data directory, you can skip **Steps 1–5** in the **How to Use** section below and start from **Step 6**.

### 📘 How to Use 📘

- **[Step 1. Build SIDs]** Run `python3 step1_sid_builder.py --dataset <dataname>`.
  - **[Detail]** This generates the PLUM-based SID for each item [(He et al., 2026)](https://dl.acm.org/doi/10.1145/3774904.3792802).

- **[Step 2. Build the SID–Item Mapping]** Run `python3 step2_sid_mapper_builder.py --dataset <dataname>`.
  - **[Detail]** This generates the mapping between each item and its corresponding SID, which is used to build subsequent training data.

- **[Step 3. Build Continued Pre-training Data]** Run `python3 step3_cpt_data_builder.py --dataset <dataname>`.
  - **[Detail]** This generates the training data for continued pre-training, which aligns the special tokens corresponding to SIDs with the LLM token space. In our experiments, we use `Qwen-3.5-4B`.

- **[Step 4. Run Continued Pre-training]** Run `bash run_cpt.sh`.
  - **[Detail 1]** This runs continued pre-training using the generated training data. The script is configured for two 80GB GPUs, so please adjust the dataset name and GPU-related settings according to your environment.
  - **[Detail 2]** In our experiments, we use a batch size of 128, with 8 samples per GPU per iteration, 8 gradient accumulation steps, and 2 GPUs.

- **[Step 5. Extract SID Tokens]** Run `python3 extract_sid_tokens.py --dataset <dataname>`.
  - **[Detail]** This extracts the aligned special token embeddings from the LLM trained in Step 4.

- **[Step 6. Build LLM Training and Evaluation Data]** Run the following three scripts:
  - **[Run 1]** `python3 step6_sid_next_item_prediction_task_builder.py --dataset <dataname>`
  - **[Run 2]** `python3 step6_sid_collaborative_neighbor_task_builder.py --dataset <dataname>`
  - **[Run 3]** `python3 step6_sid_semantic_neighbor_task_builder.py --dataset <dataname>`
  - **[Detail]** These scripts generate training and evaluation samples for the next-item prediction task, as well as training samples for the collaborative-neighbor generation and semantic-neighbor generation tasks.

- **[Step 7. Merge Data]** Run `python3 step7_merge_files.py --dataset <dataname>`.
  - **[Detail]** This merges samples from the three tasks into a single training dataset.
  - **🚨 [Key Notification] 🚨** You can control the relative weight of each task in the final loss function.

- **[Step 8. Train the LLM-based RecSys Model]** Run `bash step8_run_sid_llm.train.sh`.
  - **[Detail 1]** This trains the LLM-based generative recommender model using the constructed training dataset. The script is configured for two 80GB GPUs, so please adjust the dataset name and GPU-related settings according to your environment.
    - **[Tip]** For 40GB GPUs, we recommend processing 2 samples per GPU.
  - **[Detail 2]** In our experiments, we use a batch size of 128, with 4 samples per GPU per iteration, 16 gradient accumulation steps, and 2 GPUs.

- **[Step 9. Generate Recommendations]** Run `python3 step9_sid_evaluation_generation.py --dataset <dataname>`.
  - **[Detail]** This generates recommendations for each user. This step returns only the generated text outputs and does not directly map each recommendation to a specific item. Therefore, an additional decoding and metric calculation step is required.

- **[Step 10. Decode Outputs and Calculate Metrics]** Run `python3 step10_sid_metric_calculation.py --dataset <dataname>`.
  - **[Detail]** This decodes the generated outputs by parsing them and matching them to item IDs.