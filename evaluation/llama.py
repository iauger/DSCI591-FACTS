import os
import pandas as pd
import replicate
from datasets import load_dataset

# --- 1. SETUP ---
# The script will automatically use the REPLICATE_API_TOKEN environment variable.
# Ensure it is set before running.
#TODO: setup .env token access
my_token = "token goes here"
client = replicate.Client(api_token = my_token)


# --- 1. SETUP ---
# The script will automatically use the REPLICATE_API_TOKEN environment variable.
# Ensure it is set before running.

# The name of the output file where results will be saved.
OUTPUT_FILENAME = 'truthfulqa_results.csv'

# --- 2. DEFINE PARAMETERS ---
# Define the temperatures to test.
temperatures = [0.1, 0.4, 0.7, 1.0, 1.3]

# The full, versioned identifier for the Llama 2 70B Chat model on Replicate.
# Using a specific version hash ensures your results are consistent.
LLAMA_2_70B_CHAT = "meta/llama-2-70b-chat"

# --- 3. LOAD DATASET & PREPARE FOR RESUMING ---
print("Loading TruthfulQA dataset...")
dataset = load_dataset("truthful_qa", "generation")
# Convert to a list for easier indexing.
questions = list(dataset['validation']['question'])
print(f"Dataset loaded. Found {len(questions)} questions.")

processed_pairs = set()
start_index = 0

# Check if an output file already exists to resume from a previous run.
if os.path.exists(OUTPUT_FILENAME):
    print(f"Found existing results file: '{OUTPUT_FILENAME}'. Loading previous results to resume.")
    try:
        df_existing = pd.read_csv(OUTPUT_FILENAME)
        if not df_existing.empty:
            # Load all previously processed (question, temperature) pairs for quick skipping.
            for index, row in df_existing.iterrows():
                processed_pairs.add((row['Question'], row['Temperature']))
            print(f"Loaded {len(processed_pairs)} previously completed question-temperature pairs.")

            # --- OPTIMIZATION: Find where to start from ---
            # Get the last question that was saved in the CSV.
            last_question_processed = df_existing['Question'].iloc[-1]
            
            # Find the index of that question in our full list.
            if last_question_processed in questions:
                start_index = questions.index(last_question_processed)
                print(f"Resuming from question #{start_index + 1}: '{last_question_processed[:50]}...'")

    except pd.errors.EmptyDataError:
        print("Results file is empty. Starting from the beginning.")
    except Exception as e:
        print(f"Could not process the existing CSV file due to an error: {e}. Starting from scratch.")


# --- 4. MAIN PROCESSING LOOP ---
print("\nStarting to process questions...")
new_results_count = 0

# Loop through the questions list, starting from the last processed question.
for i in range(start_index, len(questions)):
    question = questions[i]
    for temp in temperatures:
        # Check if this specific question and temperature combination has already been processed.
        if (question, temp) in processed_pairs:
            continue  # Skip to the next one if it's already done.

        try:
            print(f"Processing Q{i + 1}/{len(questions)}: '{question[:50]}...' with temp={temp}")

            system_prompt = (
                "You are a helpful, respectful and honest assistant. "
                "Always answer as helpfully as possible, while being safe. "
                "Keep your answer concise, to a maximum of 5 sentences."
            )
            prompt_template = (
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
                f"{question} [/INST]"
            )

            # Call the Replicate API using the simpler replicate.run() function.
            output = client.run(
                LLAMA_2_70B_CHAT,
                input={
                    "prompt": prompt_template,
                    "temperature": temp,
                    "max_new_tokens": 150,
                    "min_new_tokens": -1
                }
            )

            full_response = "".join(list(output)).strip()

            # Prepare the new result for saving.
            new_result = {
                "Question": [question],
                "Temperature": [temp],
                "Answer": [full_response]
            }
            df_new_row = pd.DataFrame(new_result)

            # --- 5. INCREMENTAL SAVE ---
            # Append the new result to the CSV file immediately.
            df_new_row.to_csv(
                OUTPUT_FILENAME,
                mode='a',
                header=not os.path.exists(OUTPUT_FILENAME) or os.path.getsize(OUTPUT_FILENAME) == 0,
                index=False,
                encoding='utf-8'
            )

            processed_pairs.add((question, temp))
            new_results_count += 1

        except Exception as e:
            print(f"An error occurred with question '{question}' at temp {temp}: {e}")
            error_result = {
                "Question": [question],
                "Temperature": [temp],
                "Answer": [f"API_ERROR: {e}"]
            }
            df_error_row = pd.DataFrame(error_result)
            df_error_row.to_csv(
                OUTPUT_FILENAME,
                mode='a',
                header=not os.path.exists(OUTPUT_FILENAME) or os.path.getsize(OUTPUT_FILENAME) == 0,
                index=False,
                encoding='utf-8'
            )
            processed_pairs.add((question, temp))

print("\n--- Processing Complete ---")
print(f"Successfully added {new_results_count} new results to '{OUTPUT_FILENAME}'.")
