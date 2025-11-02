#!/bin/bash

# List of Python scripts with their directories
scripts=(
    "/home/peternicholson/Documents/A-1-a-Gsm8k/base_line_test.py"
    "/home/peternicholson/Documents/A-1-a-HpQA/base_line_HpQA_1_generate_db.py"
    "/home/peternicholson/Documents/A-1-a-HpQA/base_line_test.py"
    "/home/peternicholson/Documents/A-2-a-Gsm8k/1-A-Gsm8k.py"
    "/home/peternicholson/Documents/A-2-b-HpQA/1-A-HpQA-1.py"
    "/home/peternicholson/Documents/A-2-b-HpQA/1-A-HpQA-2.py"
    "/home/peternicholson/Documents/A-3-a-Gsm8k/1-B-Gsm8k.py"
    "/home/peternicholson/Documents/A-3-a-Gsm8k/2-Gsm8k-organise-train-eval-dataSets.py"
    "/home/peternicholson/Documents/A-3-b-HpQA/1-B-HpQA.py"
    "/home/peternicholson/Documents/A-3-b-HpQA/2-HpQA-organise-train-eval-dataSets.py"
    #----PPO train-------
    "/home/peternicholson/Documents/A-4-a-Gsm8k/1-Gsm8k-Gemma9b-it-ppo.py"
    "/home/peternicholson/Documents/A-4-a-Gsm8k/2-Gsm8k-Gemma9b-it-visualize-logs.py"
    "/home/peternicholson/Documents/A-4-b-HpQA/1-HpQA-Gemma9b-it-ppo.py"
    "/home/peternicholson/Documents/A-4-b-HpQA/2-HpQA-Gemma9b-it-visualize-logs.py"
    #----DPO train-------
    
    "/home/peternicholson/Documents/C-3-a-Gsm8k/1-generate-multiple-samples-per-prompt.py"
    "/home/peternicholson/Documents/C-3-a-Gsm8k/2-calculate-dpo-best-worst-actions.py"
    "/home/peternicholson/Documents/C-3-b-HpQA/1-generate-multiple-samples-per-prompt.py"
    "/home/peternicholson/Documents/C-3-b-HpQA/2-calculate-dpo-best-worst-actions.py"
    "/home/peternicholson/Documents/C-4-b-HpQA/1-train-dpo.py"
    "/home/peternicholson/Documents/C-4-b-HpQA/2-metrics_stats.py"
    "/home/peternicholson/Documents/C-4-a-Gsm8k/1-train-dpo.py"
    "/home/peternicholson/Documents/C-4-a-Gsm8k/2-metrics_stats.py"
    
    #run4
    "/home/peternicholson/Documents/C-3-b-HpQA/run4/0-advanced_prep.py"
    "/home/peternicholson/Documents/C-3-b-HpQA/run4/1-generate-multiple-samples-per-prompt.py"
    "/home/peternicholson/Documents/C-3-b-HpQA/run4/2-calculate-dpo-best-worst-actions.py"
    "/home/peternicholson/Documents/C-3-b-HpQA/run4/3-remove_same_best_and_worst.py"
    "/home/peternicholson/Documents/C-4-b-HpQA/run4/1-train-dpo.py"
    "/home/peternicholson/Documents/C-4-b-HpQA/run4/2-metrics_stats.py"

    #V STAR
    #'/home/peternicholson/Documents/B-2-b-HpQA/1-generate-multiple-samples-per-prompt.py'
    #'/home/peternicholson/Documents/B-2-b-HpQA/2-calculate-v-star-entries.py'
    #'/home/peternicholson/Documents/B-4-b-HpQA/1-A-PO-train.py'

    #----Evaluations------
    "/home/peternicholson/Documents/A-5-a-Gsm8k/1-Gsm8k-evaluations_test_0_shot_1000.py"
    "/home/peternicholson/Documents/A-5-a-Gsm8k/2-Gsm8k-evaluations_test_5_shot_1000.py"
    "/home/peternicholson/Documents/A-5-a-Gsm8k/3-Gsm8k-swirl-evaluations.py"
    "/home/peternicholson/Documents/A-5-b-HpQA/1-generate_db.py"
    "/home/peternicholson/Documents/A-5-b-HpQA/2-1-evaluate-recall-precision.py"
    "/home/peternicholson/Documents/A-5-b-HpQA/2-2-partial-match-answers.py"
    "/home/peternicholson/Documents/A-5-b-HpQA/3-evaluate-Swirl-cofca.py"



    #"/home/peternicholson/Documents/getHuggingFaceModel.py"
    #"/home/peternicholson/Documents/1-A-HpQA/1-A-HpQA-2.py"
    #"/home/peternicholson/Documents/1-B-HpQA/1-B-HpQA.py"
    #"/home/peternicholson/Documents/1-B-Gsm8k/1-B-Gsm8k.py"
    #"/home/peternicholson/Documents/base-Gsm8k/base_line_test.py"
    #"/home/peternicholson/Documents/base-HpQA/base_line_HpQA_1_generate_db.py"
    #"/home/peternicholson/Documents/base-HpQA/base_line_HpQA_2_test.py"
    #"/home/peternicholson/Documents/base-HpQA/base_line_HpQA_2_test_full_set.py"
    #"/home/peternicholson/Documents/base-Gsm8k/base_line_test_short_answer_train.py"
    #"/home/peternicholson/Documents/3-A-HpQA/3-A-HpQA-1-base-gemma2-27b-generate.py"
    #"/home/peternicholson/Documents/3-A-HpQA/3-A-HpQA-2-base-gemma2-27b-evaluate.py"
    #"/home/peternicholson/Documents/B-2-b-HpQA/1-generate-multiple-samples-per-prompt.py"
    #"/home/peternicholson/Documents/B-2-b-HpQA/2-calculate-v-star-entries.py"
    #"/home/peternicholson/Documents/B-2-b-2-HpQA/1-generate-multiple-samples-per-prompt.py"
    #"/home/peternicholson/Documents/B-2-b-2-HpQA/2-calculate-dpo-best-worst-actions.py"
    #"/home/peternicholson/Documents/B-4-b-2-HpQA/1-train-dpo.py"
    #"/home/peternicholson/Documents/A-4-a-Gsm8k/1-Gsm8k-organise-train-eval-dataSets.py"
    #"/home/peternicholson/Documents/A-4-a-Gsm8k/2-Gsm8k-Gemma9b-it-ppo.py"
    #"/home/peternicholson/Documents/A-4-a-Gsm8k/3-Gsm8k-Gemma9b-it-visualize-logs.py"
    #"/home/peternicholson/Documents/A-4-b-HpQA/1-HpQA-organise-train-eval-dataSets.py"
    #"/home/peternicholson/Documents/A-4-b-HpQA/2-HpQA-Gemma9b-it-ppo.py"
    #"/home/peternicholson/Documents/A-4-b-HpQA/3-HpQA-Gemma9b-it-visualize-logs.py"
    #'/home/peternicholson/Documents/A-5-b-HpQA/1-base_line_HpQA_1_generate_db.py'
    #'/home/peternicholson/Documents/A-5-b-HpQA/2-base_line_HpQA_2_test.py'
    #'/home/peternicholson/Documents/A-5-b-HpQA/3-trained_model_HpQA_2_test.py'
    #'/home/peternicholson/Documents/A-5-a-Gsm8k/1-Gsm8k-base_line_test.py'
    #'/home/peternicholson/Documents/A-5-a-Gsm8k/2-Gsm8k-trained_model_test.py'
    #'/home/peternicholson/Documents/B-4-b-HpQA/1-A-PO-train.py'
    #'/home/peternicholson/Documents/B-4-b-HpQA/3-load-model-as-inference.py'

    #run4 evaluations
    #'/home/peternicholson/Documents/A-5-b-HpQA/run4/2-1-evaluate-recall-precision.py'
    #'/home/peternicholson/Documents/A-5-b-HpQA/run4/2-2-partial-match-answers.py'
    #'/home/peternicholson/Documents/A-5-b-HpQA/run4/3-evaluate-Swirl-cofca.py'
    #'/home/peternicholson/Documents/A-5-b-HpQA/3-evaluate-Swirl-cofca.py'
)

# Loop through and run each one sequentially
for s in "${scripts[@]}"; do
    echo "Running $s..."
    python3 "$s"
    if [ $? -ne 0 ]; then
        echo "Error: $s failed. Stopping execution."
        exit 1
    fi
done

echo "All scripts completed successfully."

#chmod +x run_all.sh
#./run_all.sh
