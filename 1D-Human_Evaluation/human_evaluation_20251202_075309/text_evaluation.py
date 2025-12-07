
# text_evaluation.py
import json
import os
from datetime import datetime

def run_text_evaluation():
    """Run text-based evaluation interface"""
    print("=" * 60)
    print("FACE GENERATION QUALITY EVALUATION")
    print("=" * 60)
    
    # Get participant info
    participant_id = input("Enter participant ID (or leave blank for auto): ").strip()
    if not participant_id:
        import random
        participant_id = f"P{random.randint(1000, 9999)}"
    
    print(f"\nParticipant ID: {participant_id}")
    print("\nInstructions:")
    print("- Rate each image on scale 1-5 for each criterion")
    print("- For artifacts, enter comma-separated list")
    print("- Type 'q' to quit and save")
    
    # Load image pairs
    import glob
    image_pairs = []
    
    # This would load actual image pairs in real implementation
    # For demo, create sample pairs
    for i in range(10):
        image_pairs.append({
            'pair_id': i + 1,
            'image_a': f"ffhq_face_{i*2:04d}.png",
            'image_b': f"comparison_face_{i*2:04d}.png",
            'model_a': 'FFHQ-LDM',
            'model_b': ['StyleGAN2', 'StyleGAN3', 'Real'][i % 3]
        })
    
    ratings = []
    
    for pair in image_pairs:
        print(f"\n{'='*60}")
        print(f"IMAGE PAIR {pair['pair_id']} of {len(image_pairs)}")
        print(f"{'='*60}")
        
        print(f"\nImage A: {pair['model_a']} | Image B: {pair['model_b']}")
        print("(Images would be displayed here)")
        
        # Get ratings for Image A
        print(f"\n--- Rating Image A ---")
        a_ratings = {}
        
        for criterion in ['realism', 'symmetry', 'proportion', 'coherence', 'artifacts']:
            while True:
                try:
                    if criterion == 'artifacts':
                        artifacts = input(f"{criterion.title()} (comma-separated or 'none'): ").strip().lower()
                        if artifacts == 'q':
                            save_results(participant_id, ratings)
                            return
                        a_ratings[criterion] = artifacts if artifacts != 'none' else ''
                        break
                    else:
                        rating = input(f"{criterion.title()} (1-5): ").strip()
                        if rating == 'q':
                            save_results(participant_id, ratings)
                            return
                        rating = int(rating)
                        if 1 <= rating <= 5:
                            a_ratings[criterion] = rating
                            break
                        else:
                            print("Please enter a number between 1 and 5")
                except ValueError:
                    print("Please enter a valid number")
        
        # Get ratings for Image B
        print(f"\n--- Rating Image B ---")
        b_ratings = {}
        
        for criterion in ['realism', 'symmetry', 'proportion', 'coherence', 'artifacts']:
            while True:
                try:
                    if criterion == 'artifacts':
                        artifacts = input(f"{criterion.title()} (comma-separated or 'none'): ").strip().lower()
                        if artifacts == 'q':
                            save_results(participant_id, ratings)
                            return
                        b_ratings[criterion] = artifacts if artifacts != 'none' else ''
                        break
                    else:
                        rating = input(f"{criterion.title()} (1-5): ").strip()
                        if rating == 'q':
                            save_results(participant_id, ratings)
                            return
                        rating = int(rating)
                        if 1 <= rating <= 5:
                            b_ratings[criterion] = rating
                            break
                        else:
                            print("Please enter a number between 1 and 5")
                except ValueError:
                    print("Please enter a valid number")
        
        # Store ratings
        ratings.append({
            'pair_id': pair['pair_id'],
            'image_a': pair['image_a'],
            'image_b': pair['image_b'],
            'model_a': pair['model_a'],
            'model_b': pair['model_b'],
            'ratings_a': a_ratings,
            'ratings_b': b_ratings,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"\n✅ Pair {pair['pair_id']} completed")
    
    # Save results
    save_results(participant_id, ratings)
    print(f"\n✅ Evaluation complete! Thank you for participating.")

def save_results(participant_id, ratings):
    """Save evaluation results"""
    results = {
        'participant_id': participant_id,
        'total_pairs': len(ratings),
        'completion_time': datetime.now().isoformat(),
        'ratings': ratings
    }
    
    filename = f"evaluation_{participant_id}.json"
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    with open(os.path.join('results', filename), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: results/{filename}")

if __name__ == "__main__":
    run_text_evaluation()
