from huggingface_hub import HfApi, ModelCard, ModelCardData
from huggingface_hub.utils import RepositoryNotFoundError
import concurrent.futures
from tqdm import tqdm
import time
import argparse  # For CLI argument parsing

# Initialize HF API
api = HfApi()

def get_models_with_name(name):
    """Fetch all models containing the given name."""
    return list(api.list_models(search=name))

def create_pr_for_model(model_id, changes, pr_description):
    """Create a PR for a model with the specified changes and description."""
    try:
        # Get current model card
        card = ModelCard.load(model_id)
        
        # Prepare changes
        card_data = card.data.to_dict()
        for key, value in changes.items():
            if key == 'tags':
                current_tags = card_data.get('tags', [])
                if value not in current_tags:
                    current_tags.append(value)
                    card_data['tags'] = current_tags
            elif key == 'pipeline_tag' and 'pipeline_tag' not in card_data:
                card_data['pipeline_tag'] = value
            elif key == 'base_model':
                card_data['base_model'] = value
        
        # Update card
        updated_card = ModelCard.from_template(
            ModelCardData(**card_data),
            template_str=card.content  # Keep existing content
        )
        
        # Create PR with description
        updated_card.push_to_hub(
            model_id,
            commit_message="Add Robotics tag and metadata",
            pr_description=pr_description,
            create_pr=True,
            repo_type="model"
        )
        return True
    except Exception as e:
        print(f"Error processing {model_id}: {str(e)}")
        return False

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for model in all_models:
            model_id = model if use_direct_ids else model.modelId
            futures.append(executor.submit(create_pr_for_model, model_id, changes, pr_description))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=batch_name):
            if future.result():
                success_count += 1
            time.sleep(1)  # Rate limiting
    
    print(f"Successfully created PRs for {success_count}/{len(all_models)} models in {batch_name}")

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
