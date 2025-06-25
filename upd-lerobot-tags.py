from huggingface_hub import HfApi, ModelCard, ModelCardData
from huggingface_hub.utils import RepositoryNotFoundError
import concurrent.futures
from tqdm import tqdm
import time
import argparse  # For CLI argument parsing
import textwrap  # For generating README templates
import requests  # To catch HTTP errors from ModelCard.load

# Initialize HF API
api = HfApi()

def get_models_with_name(name):
    """Fetch all models containing the given name."""
    return list(api.list_models(search=name))

def generate_readme_batch1(model_id: str) -> str:
    """Generate a default README.md content for Batch 1 repositories."""
    model_name = model_id.split("/")[-1]
    return textwrap.dedent(f"""\
    ---
    library_name: lerobot
    license: apache-2.0
    pipeline_tag: robotics
    tags:
    - robotics
    ---

    # Model Card for {model_name}

    <!-- Provide a quick summary of what the model is/does. -->


    This policy has been trained and pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot).
    See the full documentation at [LeRobot Docs](https://huggingface.co/docs/lerobot/index).

    ---

    ## How to Get Started with the Model

    For a complete walkthrough, see the [training guide](https://huggingface.co/docs/lerobot/il_robots#train-a-policy).
    Below is the short version on how to train and run inference/eval:

    ### Train from scratch

    ```bash
    python lerobot/scripts/train.py \
        --dataset.repo_id=<user _or_org>/<dataset> \
        --policy.type=act \
        --output_dir=outputs/train/<desired_policy_repo_id> \
        --job_name=lerobot_training \
        --policy.device=cuda \
        --policy.repo_id=<user_or_org>/<desired_policy_repo_id> \
        --wandb.enable=true
    ```

    *Writes checkpoints to `outputs/train/<desired_policy_repo_id>/checkpoints/`.*

    ### Evaluate the policy

    ```bash
    python -m lerobot.record \
        --robot.type=so100_follower \
        --dataset.repo_id=<user_or_org>/eval_<dataset> \
        --policy.path=<user_or_org>/<desired_policy_repo_id> \
        --episodes=10
    ```

    Prefix the dataset repo with **eval_** and supply `--policy.path` pointing to a local or hub checkpoint.

    ---
    """)

def generate_readme_batch2(model_id: str, base_model_name: str) -> str:
    """Generate a default README.md content for Batch 2 repositories."""
    repo_name = model_id.split("/")[-1]
    base_model_repo = f"lerobot/{base_model_name}_base"
    return textwrap.dedent(f"""---
base_model: {base_model_repo}
library_name: lerobot
license: apache-2.0
model_name: {base_model_name}
pipeline_tag: robotics
tags:
- robotics
- {base_model_name}
---

# Model Card for {repo_name}

<!-- Provide a quick summary of what the model is/does. -->


[SmolVLA](https://huggingface.co/papers/2506.01844) is a compact, efficient vision-language-action model that achieves competitive performance at reduced computational costs and can be deployed on consumer-grade hardware.


This policy has been trained and pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot).
See the full documentation at [LeRobot Docs](https://huggingface.co/docs/lerobot/index).

---

## How to Get Started with the Model

For a complete walkthrough, see the [training guide](https://huggingface.co/docs/lerobot/il_robots#train-a-policy).
Below is the short version on how to train and run inference/eval:

### Train from scratch

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=<user_or_org>/<dataset> \
  --policy.type=act \
  --output_dir=outputs/train/<desired_policy_repo_id> \
  --job_name=lerobot_training \
  --policy.device=cuda \
  --policy.repo_id=<user_or_org>/<desired_policy_repo_id> \
  --wandb.enable=true
```

*Writes checkpoints to `outputs/train/<desired_policy_repo_id>/checkpoints/`.*

### Evaluate the policy

```bash
python -m lerobot.record \
  --robot.type=so100_follower \
  --dataset.repo_id=<user_or_org>/eval_<dataset> \
  --policy.path=<user_or_org>/<desired_policy_repo_id> \
  --episodes=10
```

Prefix the dataset repo with **eval_** and supply `--policy.path` pointing to a local or hub checkpoint.

---
""")

def create_pr_for_model(model_id, changes, pr_description):
    """Create a PR for a model with the specified changes and description."""
    try:
        # Try to load existing model card; if README is missing, continue with blank template
        try:
            card = ModelCard.load(model_id)
            card_data = card.data.to_dict()
            card_content = card.content
        except (requests.exceptions.HTTPError, RepositoryNotFoundError):
            # README.md not found – start with empty card
            card = None
            card_data = {}
            card_content = ""

        # Prepare changes
        for key, value in changes.items():
            if key == 'tags':
                current_tags = card_data.get('tags', [])
                if value not in current_tags:
                    current_tags.append(value)
                    card_data['tags'] = current_tags
            elif key == 'pipeline_tag' and 'pipeline_tag' not in card_data:
                card_data['pipeline_tag'] = value
            elif key == 'base_model':
                # Store full repo path for base model for clarity
                full_base_model = value if value.startswith("lerobot/") else f"lerobot/{value}_base"
                card_data['base_model'] = full_base_model
                # Also add the base model name as an extra tag for discoverability
                extra_tag = value.split("/")[-1]  # get plain name without namespace
                current_tags = card_data.get('tags', [])
                if extra_tag not in current_tags:
                    current_tags.append(extra_tag)
                    card_data['tags'] = current_tags
        
        # Determine which template to use when README is missing
        if not card_content.strip():
            if 'base_model' in changes:
                template_str = generate_readme_batch2(model_id, changes['base_model'])
            else:
                template_str = generate_readme_batch1(model_id)
        else:
            template_str = card_content
        
        # Update card
        updated_card = ModelCard.from_template(
            ModelCardData(**card_data),
            template_str=template_str  # Use existing or newly generated content
        )
        
        # Create PR with description
        pr_url = updated_card.push_to_hub(
            model_id,
            commit_message="Add Robotics tag and metadata",
            commit_description=pr_description,
            create_pr=True,
            repo_type="model"
        )
        return pr_url
    except Exception as e:
        print(f"Error processing {model_id}: {str(e)}")
        return None

def batch_process_models(model_names, changes, batch_name, pr_description, use_direct_ids=False):
    """Process a batch of models with the given changes and PR description."""
    all_models = []
    if use_direct_ids:
        # Treat provided names as full repository IDs
        all_models = model_names
    else:
        for name in model_names:
            all_models.extend(get_models_with_name(name))
    
    print(f"Found {len(all_models)} models for {batch_name}")
    
    # Process with threading (limited to 5 concurrent requests to be polite)
    success_count = 0
    pr_links = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for model in all_models:
            model_id = model if use_direct_ids else model.modelId
            futures.append(executor.submit(create_pr_for_model, model_id, changes, pr_description))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=batch_name):
            pr_url = future.result()
            if pr_url:
                success_count += 1
                pr_links.append(pr_url)
            time.sleep(1)  # Rate limiting
    
    print(f"Successfully created PRs for {success_count}/{len(all_models)} models in {batch_name}")
    if pr_links:
        print("Created Pull Requests:")
        for link in pr_links:
            print(f" - {link}")

def main():
    parser = argparse.ArgumentParser(description="Bulk update robotics metadata on Hugging Face models.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode; model names are treated as full repo IDs.")
    parser.add_argument("--batch1", nargs="+", default=[], help="Model repo IDs to process in Batch 1 when --debug is used.")
    parser.add_argument("--batch2", nargs="+", default=[], help="Model repo IDs to process in Batch 2 when --debug is used.")

    args = parser.parse_args()

    # PR descriptions
    batch1_pr_description = """This PR adds standard Robotics metadata to the model card:
- Added 'Robotics' to tags
- Set pipeline_tag to 'robotics'

These changes help improve discoverability of robotics-related models on the Hugging Face Hub.
"""
    
    batch2_pr_description = """This PR adds standard Robotics metadata and base model information:
- Added 'Robotics' to tags
- Set pipeline_tag to 'robotics'
- Added 'smolvla' as base_model

These changes help improve discoverability and provide better model lineage information.
"""

    # Batch 1: Add Robotics tag/pipeline_tag
    batch1_names = ['so101', 'so100', 'lerobot', 'vqbet', 'pi0', 'pi0fast']
    batch1_changes = {
        'tags': 'Robotics',
        'pipeline_tag': 'robotics'
    }
    
    # Batch 2: Add Robotics tag + smolvla as base_model
    batch2_names = ['smolvla']
    batch2_changes = {
        'tags': 'Robotics',
        'pipeline_tag': 'robotics',
        'base_model': 'smolvla'
    }

    # Override default names when debug mode is active and lists provided
    direct_ids = False
    if args.debug:
        if args.batch1:
            batch1_names = args.batch1
        if args.batch2:
            batch2_names = args.batch2
        direct_ids = True  # Provided names are full repo IDs
    
    print("Starting batch processing...")
    batch_process_models(batch1_names, batch1_changes, "Batch 1 (Add Robotics tag)", batch1_pr_description, use_direct_ids=direct_ids)
    batch_process_models(batch2_names, batch2_changes, "Batch 2 (Add Robotics tag + base_model)", batch2_pr_description, use_direct_ids=direct_ids)
    
    print("All batches processed!")

if __name__ == "__main__":
    main()
