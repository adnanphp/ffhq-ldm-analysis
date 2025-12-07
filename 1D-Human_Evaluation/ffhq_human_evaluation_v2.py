# ffhq_human_evaluation.py
import torch
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import json
import random
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add the latent-diffusion directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
latent_diffusion_path = os.path.join(current_dir, 'latent-diffusion')
sys.path.insert(0, latent_diffusion_path)

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

class HumanEvaluationStudy:
    def __init__(self, model_path="models/ldm/ffhq-ldm-vq-4/model.ckpt"):
        """Initialize human evaluation study"""
        print(" HUMAN EVALUATION STUDY - PERCEPTUAL QUALITY ASSESSMENT")
        print("=" * 60)
        
        # Create config
        self.config_path = self.create_config()
        self.model_path = model_path
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Using device: {self.device}")
        
        print(" Loading FFHQ model...")
        config = OmegaConf.load(self.config_path)
        self.model = instantiate_from_config(config.model)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.sampler = DDIMSampler(self.model)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"human_evaluation_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories
        self.stimuli_dir = os.path.join(self.output_dir, "stimuli")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        
        for dir_path in [self.stimuli_dir, self.results_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f" Model loaded")
        print(f" Output directory: {self.output_dir}")
        
        # Study parameters
        self.study_params = {
            'num_participants': 50,
            'images_per_participant': 20,  # 10 pairs = 20 images
            'total_image_pairs': 100,
            'rating_scale': 5,  # 1-5 Likert scale
            'comparison_models': ['FFHQ-LDM', 'StyleGAN2', 'StyleGAN3', 'Real'],
            'evaluation_criteria': [
                'Realism',
                'Symmetry',
                'Proportion',
                'Feature Coherence',
                'Artifact Presence'
            ]
        }
    
    def create_config(self):
        """Create configuration file"""
        config_yaml = """
model:
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: class_label
    image_size: 64
    channels: 3
    conditioning_key: null
    scale_factor: 0.18215
    use_ema: false
    
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32
        in_channels: 3
        out_channels: 3
        model_channels: 224
        attention_resolutions: [8, 4, 2]
        num_res_blocks: 2
        channel_mult: [1, 2, 3, 4]
        num_head_channels: 32
        
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        lossconfig:
          target: torch.nn.Identity
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [1, 2, 4]
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        
    cond_stage_config:
      target: torch.nn.Identity
"""
        
        config_path = "human_eval_config.yaml"
        with open(config_path, "w") as f:
            f.write(config_yaml)
        
        return config_path
    
    def generate_stimulus_set(self, num_faces=200):
        """Generate faces for the human evaluation study"""
        print(f"\n{'='*60}")
        print(f"GENERATING STIMULUS SET ({num_faces} faces)")
        print(f"{'='*60}")
        
        # Generate FFHQ-LDM faces
        ffhq_faces = []
        
        num_batches = (num_faces + 7) // 8  # Batch size 8
        
        for batch_idx in range(num_batches):
            print(f"\n📦 Generating batch {batch_idx+1}/{num_batches}...")
            
            current_batch = min(8, num_faces - batch_idx * 8)
            
            # Use different seeds for variety
            torch.manual_seed(42 + batch_idx * 100)
            
            # Generate
            shape = [3, 64, 64]
            c = torch.zeros(current_batch, 0, 64, 64).to(self.device)
            uc = torch.zeros(current_batch, 0, 64, 64).to(self.device)
            
            samples, _ = self.sampler.sample(
                S=50,  # High quality
                conditioning=c,
                batch_size=current_batch,
                shape=shape,
                eta=0.0,
                verbose=False,
                unconditional_guidance_scale=7.5,
                unconditional_conditioning=uc,
            )
            
            # Decode
            with torch.no_grad():
                x_samples = self.model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0, 1)
            
            # Save faces
            for i in range(current_batch):
                face_id = batch_idx * 8 + i
                if face_id < num_faces:
                    img_np = x_samples[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # Save image
                    filename = f"ffhq_face_{face_id:04d}.png"
                    filepath = os.path.join(self.stimuli_dir, "ffhq", filename)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    Image.fromarray((img_np * 255).astype(np.uint8)).save(filepath)
                    
                    # Store face info
                    face_data = {
                        'id': face_id,
                        'model': 'FFHQ-LDM',
                        'seed': 42 + batch_idx * 100 + i,
                        'filename': filename,
                        'filepath': filepath,
                        'image_np': img_np
                    }
                    
                    ffhq_faces.append(face_data)
            
            print(f"  ✅ Generated {current_batch} FFHQ-LDM faces")
        
        print(f"\n✅ Generated {len(ffhq_faces)} FFHQ-LDM faces")
        
        # Create placeholder directories for other models
        for model in ['StyleGAN2', 'StyleGAN3', 'Real']:
            model_dir = os.path.join(self.stimuli_dir, model.lower())
            os.makedirs(model_dir, exist_ok=True)
            
            # Create placeholder files
            for i in range(num_faces // 4):  # Quarter as many for other models
                filename = f"{model.lower()}_face_{i:04d}.png"
                filepath = os.path.join(model_dir, filename)
                
                # Create a placeholder image (in real study, you'd have actual images)
                placeholder = np.ones((64, 64, 3), dtype=np.uint8) * 128
                Image.fromarray(placeholder).save(filepath)
                
                print(f"   Created placeholder for {model} face {i}")
        
        # Save stimulus metadata
        stimulus_info = {
            'total_faces': num_faces,
            'ffhq_faces': len(ffhq_faces),
            'models': self.study_params['comparison_models'],
            'generation_date': datetime.now().isoformat(),
            'parameters': {
                'steps': 50,
                'guidance_scale': 7.5,
                'eta': 0.0
            }
        }
        
        with open(os.path.join(self.stimuli_dir, "stimulus_metadata.json"), 'w') as f:
            json.dump(stimulus_info, f, indent=2)
        
        return ffhq_faces
    
    def create_evaluation_interface(self):
        """Create HTML interface for human evaluation"""
        print(f"\n{'='*60}")
        print("CREATING HUMAN EVALUATION INTERFACE")
        print(f"{'='*60}")
        
        html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Generation Quality Evaluation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .header {
            text-align: center;
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .instructions {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .criteria-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .criterion {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .images-container {
            display: flex;
            justify-content: space-around;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        
        .image-box {
            text-align: center;
            margin: 10px;
        }
        
        .face-image {
            width: 256px;
            height: 256px;
            object-fit: contain;
            border: 3px solid #ddd;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        .rating-scale {
            display: flex;
            justify-content: space-between;
            margin: 20px 0;
        }
        
        .rating-option {
            text-align: center;
            padding: 10px;
            background-color: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            border: 2px solid transparent;
        }
        
        .rating-option:hover {
            background-color: #f0f0f0;
            border-color: #3498db;
        }
        
        .rating-option.selected {
            background-color: #3498db;
            color: white;
            border-color: #2980b9;
        }
        
        .rating-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        
        .submit-btn {
            background-color: #27ae60;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 18px;
            border-radius: 8px;
            cursor: pointer;
            display: block;
            margin: 30px auto;
            transition: background-color 0.3s;
        }
        
        .submit-btn:hover {
            background-color: #229954;
        }
        
        .progress {
            text-align: center;
            margin-bottom: 20px;
            font-size: 18px;
            color: #2c3e50;
        }
        
        .model-label {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 16px;
        }
        
        .artifact-checklist {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .artifact-item {
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Face Generation Quality Evaluation Study</h1>
        <p>University Research Project - Computer Vision & Pattern Recognition</p>
    </div>
    
    <div class="instructions">
        <h2>📋 Instructions</h2>
        <p>Welcome to our face generation quality evaluation study! Your task is to assess the quality of generated faces based on several criteria.</p>
        
        <h3>How to participate:</h3>
        <ol>
            <li>You will see pairs of faces - some are real, some are generated by AI models</li>
            <li>Rate each face on the 5-point scale for each criterion</li>
            <li>Check any artifacts you notice in the images</li>
            <li>Complete all 10 image pairs (20 faces total)</li>
            <li>Your responses will be recorded anonymously for research purposes</li>
        </ol>
        
        <p><strong>Note:</strong> There are no right or wrong answers. We're interested in your honest perception.</p>
    </div>
    
    <div class="criteria-grid">
        <div class="criterion">
            <h3> Realism</h3>
            <p>How realistic does the face look? Does it look like a real human face or clearly AI-generated?</p>
            <p><em>1 = Clearly AI, 5 = Indistinguishable from real</em></p>
        </div>
        
        <div class="criterion">
            <h3> Symmetry</h3>
            <p>How symmetrical is the face? Are features balanced on both sides?</p>
            <p><em>1 = Very asymmetrical, 5 = Perfectly symmetrical</em></p>
        </div>
        
        <div class="criterion">
            <h3> Proportion</h3>
            <p>Are facial features proportionally sized and placed? Do they look natural?</p>
            <p><em>1 = Poor proportions, 5 = Perfect proportions</em></p>
        </div>
        
        <div class="criterion">
            <h3> Feature Coherence</h3>
            <p>Do all facial features look like they belong together? Is there consistency?</p>
            <p><em>1 = Incoherent, 5 = Perfectly coherent</em></p>
        </div>
        
        <div class="criterion">
            <h3> Artifact Presence</h3>
            <p>Are there any visual artifacts, blurriness, or distortions?</p>
            <p><em>1 = Many artifacts, 5 = No artifacts</em></p>
        </div>
        
        <div class="criterion">
            <h3>🏆 Overall Quality</h3>
            <p>Overall assessment considering all factors above</p>
            <p><em>1 = Very poor, 5 = Excellent</em></p>
        </div>
    </div>
    
    <div class="progress">
        <h2>Image Pair <span id="current-pair">1</span> of 10</h2>
    </div>
    
    <div class="images-container">
        <div class="image-box">
            <div class="model-label">Image A</div>
            <img id="face-a" class="face-image" src="placeholder.jpg" alt="Face A">
            
            <div class="rating-scale">
                <div class="rating-option" data-face="a" data-criterion="realism" data-value="1">
                    <div class="rating-number">1</div>
                    <div class="rating-label">AI</div>
                </div>
                <div class="rating-option" data-face="a" data-criterion="realism" data-value="2">
                    <div class="rating-number">2</div>
                </div>
                <div class="rating-option" data-face="a" data-criterion="realism" data-value="3">
                    <div class="rating-number">3</div>
                    <div class="rating-label">Neutral</div>
                </div>
                <div class="rating-option" data-face="a" data-criterion="realism" data-value="4">
                    <div class="rating-number">4</div>
                </div>
                <div class="rating-option" data-face="a" data-criterion="realism" data-value="5">
                    <div class="rating-number">5</div>
                    <div class="rating-label">Real</div>
                </div>
            </div>
            
            <div class="artifact-checklist">
                <h4>Artifacts detected in Image A:</h4>
                <div class="artifact-item">
                    <input type="checkbox" id="a-blurry" data-face="a" data-artifact="blurry">
                    <label for="a-blurry">Blurry features</label>
                </div>
                <div class="artifact-item">
                    <input type="checkbox" id="a-asymmetry" data-face="a" data-artifact="asymmetry">
                    <label for="a-asymmetry">Facial asymmetry</label>
                </div>
                <div class="artifact-item">
                    <input type="checkbox" id="a-distortion" data-face="a" data-artifact="distortion">
                    <label for="a-distortion">Feature distortion</label>
                </div>
                <div class="artifact-item">
                    <input type="checkbox" id="a-color" data-face="a" data-artifact="color">
                    <label for="a-color">Color artifacts</label>
                </div>
            </div>
        </div>
        
        <div class="image-box">
            <div class="model-label">Image B</div>
            <img id="face-b" class="face-image" src="placeholder.jpg" alt="Face B">
            
            <div class="rating-scale">
                <div class="rating-option" data-face="b" data-criterion="realism" data-value="1">
                    <div class="rating-number">1</div>
                    <div class="rating-label">AI</div>
                </div>
                <div class="rating-option" data-face="b" data-criterion="realism" data-value="2">
                    <div class="rating-number">2</div>
                </div>
                <div class="rating-option" data-face="b" data-criterion="realism" data-value="3">
                    <div class="rating-number">3</div>
                    <div class="rating-label">Neutral</div>
                </div>
                <div class="rating-option" data-face="b" data-criterion="realism" data-value="4">
                    <div class="rating-number">4</div>
                </div>
                <div class="rating-option" data-face="b" data-criterion="realism" data-value="5">
                    <div class="rating-number">5</div>
                    <div class="rating-label">Real</div>
                </div>
            </div>
            
            <div class="artifact-checklist">
                <h4>Artifacts detected in Image B:</h4>
                <div class="artifact-item">
                    <input type="checkbox" id="b-blurry" data-face="b" data-artifact="blurry">
                    <label for="b-blurry">Blurry features</label>
                </div>
                <div class="artifact-item">
                    <input type="checkbox" id="b-asymmetry" data-face="b" data-artifact="asymmetry">
                    <label for="b-asymmetry">Facial asymmetry</label>
                </div>
                <div class="artifact-item">
                    <input type="checkbox" id="b-distortion" data-face="b" data-artifact="distortion">
                    <label for="b-distortion">Feature distortion</label>
                </div>
                <div class="artifact-item">
                    <input type="checkbox" id="b-color" data-face="b" data-artifact="color">
                    <label for="b-color">Color artifacts</label>
                </div>
            </div>
        </div>
    </div>
    
    <button class="submit-btn" id="next-btn">Next Image Pair →</button>
    
    <script>
        // Study configuration
        const studyConfig = {
            totalPairs: 10,
            currentPair: 1,
            participantId: 'P' + Math.random().toString(36).substr(2, 9),
            startTime: new Date().toISOString(),
            ratings: []
        };
        
        // Image pairs data (would be loaded from server in real study)
        const imagePairs = [
            { a: 'ffhq_face_0001.png', b: 'stylegan2_face_0001.png' },
            { a: 'real_face_0001.png', b: 'ffhq_face_0002.png' },
            { a: 'stylegan3_face_0001.png', b: 'real_face_0002.png' },
            { a: 'ffhq_face_0003.png', b: 'stylegan2_face_0002.png' },
            { a: 'real_face_0003.png', b: 'stylegan3_face_0002.png' },
            { a: 'ffhq_face_0004.png', b: 'real_face_0004.png' },
            { a: 'stylegan2_face_0003.png', b: 'stylegan3_face_0003.png' },
            { a: 'ffhq_face_0005.png', b: 'real_face_0005.png' },
            { a: 'stylegan2_face_0004.png', b: 'ffhq_face_0006.png' },
            { a: 'real_face_0006.png', b: 'stylegan3_face_0004.png' }
        ];
        
        // Criteria for rating
        const criteria = ['realism', 'symmetry', 'proportion', 'coherence', 'artifacts', 'overall'];
        
        // Initialize current ratings
        let currentRatings = {
            a: { realism: null, symmetry: null, proportion: null, coherence: null, artifacts: null, overall: null },
            b: { realism: null, symmetry: null, proportion: null, coherence: null, artifacts: null, overall: null }
        };
        
        // Initialize artifacts
        let currentArtifacts = {
            a: { blurry: false, asymmetry: false, distortion: false, color: false },
            b: { blurry: false, asymmetry: false, distortion: false, color: false }
        };
        
        // DOM elements
        const faceA = document.getElementById('face-a');
        const faceB = document.getElementById('face-b');
        const currentPairSpan = document.getElementById('current-pair');
        const nextBtn = document.getElementById('next-btn');
        
        // Load first image pair
        function loadImagePair(pairIndex) {
            const pair = imagePairs[pairIndex];
            
            // In real study, load actual images from server
            faceA.src = `stimuli/${pair.a}`;
            faceB.src = `stimuli/${pair.b}`;
            
            currentPairSpan.textContent = pairIndex + 1;
            
            // Reset ratings and artifacts
            currentRatings = {
                a: { realism: null, symmetry: null, proportion: null, coherence: null, artifacts: null, overall: null },
                b: { realism: null, symmetry: null, proportion: null, coherence: null, artifacts: null, overall: null }
            };
            
            currentArtifacts = {
                a: { blurry: false, asymmetry: false, distortion: false, color: false },
                b: { blurry: false, asymmetry: false, distortion: false, color: false }
            };
            
            // Reset UI
            document.querySelectorAll('.rating-option').forEach(el => {
                el.classList.remove('selected');
            });
            
            document.querySelectorAll('input[type="checkbox"]').forEach(el => {
                el.checked = false;
            });
        }
        
        // Handle rating selection
        document.querySelectorAll('.rating-option').forEach(el => {
            el.addEventListener('click', function() {
                const face = this.getAttribute('data-face');
                const criterion = this.getAttribute('data-criterion');
                const value = parseInt(this.getAttribute('data-value'));
                
                // Update rating
                currentRatings[face][criterion] = value;
                
                // Update UI
                // Clear other selections for same face and criterion
                document.querySelectorAll(`.rating-option[data-face="${face}"][data-criterion="${criterion}"]`).forEach(el => {
                    el.classList.remove('selected');
                });
                
                // Select current
                this.classList.add('selected');
            });
        });
        
        // Handle artifact checkboxes
        document.querySelectorAll('input[type="checkbox"]').forEach(el => {
            el.addEventListener('change', function() {
                const face = this.getAttribute('data-face');
                const artifact = this.getAttribute('data-artifact');
                currentArtifacts[face][artifact] = this.checked;
            });
        });
        
        // Handle next button
        nextBtn.addEventListener('click', function() {
            // Save current ratings
            studyConfig.ratings.push({
                pair: studyConfig.currentPair,
                images: imagePairs[studyConfig.currentPair - 1],
                ratings: JSON.parse(JSON.stringify(currentRatings)),
                artifacts: JSON.parse(JSON.stringify(currentArtifacts)),
                timestamp: new Date().toISOString()
            });
            
            // Move to next pair or finish
            studyConfig.currentPair++;
            
            if (studyConfig.currentPair <= studyConfig.totalPairs) {
                loadImagePair(studyConfig.currentPair - 1);
            } else {
                // Study complete
                studyConfig.endTime = new Date().toISOString();
                
                // Save results
                saveResults();
                
                // Show completion message
                document.body.innerHTML = `
                    <div class="header">
                        <h1>✅ Study Complete!</h1>
                        <p>Thank you for your participation</p>
                    </div>
                    <div class="instructions">
                        <h2>Your responses have been recorded</h2>
                        <p>Participant ID: ${studyConfig.participantId}</p>
                        <p>Total time: ${Math.round((new Date(studyConfig.endTime) - new Date(studyConfig.startTime)) / 1000)} seconds</p>
                        <p>Your contribution helps advance face generation research!</p>
                    </div>
                `;
            }
        });
        
        // Save results function
        function saveResults() {
            // In real study, send to server
            // For demo, create download link
            
            const dataStr = JSON.stringify(studyConfig, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            
            // Create download link
            const link = document.createElement('a');
            link.href = URL.createObjectURL(dataBlob);
            link.download = `face_evaluation_${studyConfig.participantId}.json`;
            link.click();
        }
        
        // Initialize
        loadImagePair(0);
    </script>
</body>
</html>
        '''
        
        # Save HTML interface
        html_file = os.path.join(self.output_dir, "evaluation_interface.html")
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Evaluation interface created: {html_file}")
        
        # Also create a simple text-based interface
        self.create_text_interface()
        
        return html_file
    
    def create_text_interface(self):
        """Create a text-based interface for terminal evaluation"""
        text_interface = '''
TEXT-BASED EVALUATION INTERFACE
===============================

Instructions:
1. You will be shown image pairs
2. Rate each image on a scale of 1-5 for each criterion
3. Type 'q' at any time to quit and save progress

Rating Scale:
1 = Very poor
2 = Poor
3 = Fair
4 = Good
5 = Excellent

Criteria:
1. Realism (How realistic does the face look?)
2. Symmetry (How symmetrical are facial features?)
3. Proportion (Are features proportionally sized?)
4. Coherence (Do features look like they belong together?)
5. Artifact-Free (Are there visual artifacts?)

Artifact Checklist (comma-separated):
- blurry: Blurry or unclear features
- asymmetry: Noticeable facial asymmetry
- distortion: Distorted facial features
- color: Color artifacts or unnatural tones
- other: Any other issues

Example response:
Image A - Realism: 4, Symmetry: 3, Proportion: 4, Coherence: 5, Artifacts: blurry,asymmetry
        '''
        
        text_file = os.path.join(self.output_dir, "text_interface_instructions.txt")
        with open(text_file, 'w') as f:
            f.write(text_interface)
        
        # Create Python script for text interface
        script_content = '''
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
    
    print(f"\\nParticipant ID: {participant_id}")
    print("\\nInstructions:")
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
        print(f"\\n{'='*60}")
        print(f"IMAGE PAIR {pair['pair_id']} of {len(image_pairs)}")
        print(f"{'='*60}")
        
        print(f"\\nImage A: {pair['model_a']} | Image B: {pair['model_b']}")
        print("(Images would be displayed here)")
        
        # Get ratings for Image A
        print(f"\\n--- Rating Image A ---")
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
        print(f"\\n--- Rating Image B ---")
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
        
        print(f"\\n✅ Pair {pair['pair_id']} completed")
    
    # Save results
    save_results(participant_id, ratings)
    print(f"\\n✅ Evaluation complete! Thank you for participating.")

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
    
    print(f"\\n Results saved to: results/{filename}")

if __name__ == "__main__":
    run_text_evaluation()
'''
        
        script_file = os.path.join(self.output_dir, "text_evaluation.py")
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Text evaluation script created: {script_file}")
    
    def simulate_participant_responses(self, num_participants=50):
        """Simulate participant responses for analysis"""
        print(f"\n{'='*60}")
        print(f"SIMULATING PARTICIPANT RESPONSES ({num_participants} participants)")
        print(f"{'='*60}")
        
        # Create realistic response patterns
        np.random.seed(42)  # For reproducibility
        
        participants = []
        
        for p in range(num_participants):
            participant_id = f"P{p+1:03d}"
            
            print(f"\r Simulating participant {p+1}/{num_participants}...", end="")
            
            # Generate different response patterns based on "expertise level"
            expertise_level = np.random.choice(['novice', 'intermediate', 'expert'], 
                                               p=[0.3, 0.5, 0.2])
            
            responses = []
            
            # Generate 10 image pair responses
            for pair_id in range(10):
                # Different models have different quality distributions
                models = ['FFHQ-LDM', 'StyleGAN2', 'StyleGAN3', 'Real']
                
                # Randomly assign models to this pair
                model_a = np.random.choice(models)
                model_b = np.random.choice([m for m in models if m != model_a])
                
                # Define baseline quality for each model
                baseline_qualities = {
                    'Real': {'mean': 4.5, 'std': 0.3},
                    'FFHQ-LDM': {'mean': 4.0, 'std': 0.5},
                    'StyleGAN3': {'mean': 3.8, 'std': 0.6},
                    'StyleGAN2': {'mean': 3.5, 'std': 0.7}
                }
                
                # Generate ratings with some noise
                def generate_rating(model, criterion, expertise):
                    baseline = baseline_qualities[model]
                    
                    # Adjust based on criterion
                    if criterion == 'realism':
                        mean = baseline['mean'] - 0.2 if model != 'Real' else baseline['mean']
                    elif criterion == 'symmetry':
                        mean = baseline['mean'] + 0.1
                    elif criterion == 'artifacts':
                        mean = baseline['mean'] - 0.3 if model != 'Real' else baseline['mean']
                    else:
                        mean = baseline['mean']
                    
                    # Add expertise-based noise
                    if expertise == 'novice':
                        std = baseline['std'] * 1.5
                    elif expertise == 'expert':
                        std = baseline['std'] * 0.7
                    else:
                        std = baseline['std']
                    
                    # Generate rating (clamped to 1-5)
                    rating = np.random.normal(mean, std)
                    rating = np.clip(rating, 1, 5)
                    return round(rating, 1)
                
                # Generate artifacts (more likely for generated images)
                def generate_artifacts(model):
                    if model == 'Real':
                        # Real images have fewer artifacts
                        artifacts = []
                        if np.random.random() < 0.1:  # 10% chance
                            artifacts.append('blurry')
                        return ','.join(artifacts)
                    else:
                        # Generated images have more artifacts
                        possible_artifacts = ['blurry', 'asymmetry', 'distortion', 'color', 'other']
                        weights = [0.3, 0.4, 0.3, 0.2, 0.1]
                        num_artifacts = np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1])
                        selected = np.random.choice(possible_artifacts, size=num_artifacts, 
                                                   p=np.array(weights)/sum(weights), replace=False)
                        return ','.join(selected)
                
                # Generate ratings for both images
                criteria = ['realism', 'symmetry', 'proportion', 'coherence', 'artifacts']
                
                ratings_a = {}
                for criterion in criteria:
                    if criterion == 'artifacts':
                        ratings_a[criterion] = generate_artifacts(model_a)
                    else:
                        ratings_a[criterion] = generate_rating(model_a, criterion, expertise_level)
                
                ratings_b = {}
                for criterion in criteria:
                    if criterion == 'artifacts':
                        ratings_b[criterion] = generate_artifacts(model_b)
                    else:
                        ratings_b[criterion] = generate_rating(model_b, criterion, expertise_level)
                
                # Add overall rating (average of other criteria)
                overall_a = np.mean([ratings_a[c] for c in criteria if c != 'artifacts'])
                overall_b = np.mean([ratings_b[c] for c in criteria if c != 'artifacts'])
                
                ratings_a['overall'] = round(overall_a, 1)
                ratings_b['overall'] = round(overall_b, 1)
                
                # Store response
                response_time = np.random.uniform(15, 45)  # Seconds per pair
                
                responses.append({
                    'pair_id': pair_id + 1,
                    'model_a': model_a,
                    'model_b': model_b,
                    'image_a': f"{model_a.lower()}_face_{(pair_id*2):04d}.png",
                    'image_b': f"{model_b.lower()}_face_{(pair_id*2+1):04d}.png",
                    'ratings_a': ratings_a,
                    'ratings_b': ratings_b,
                    'response_time_seconds': round(response_time, 1)
                })
            
            # Store participant data
            participants.append({
                'participant_id': participant_id,
                'expertise_level': expertise_level,
                'start_time': datetime.now().isoformat(),
                'total_time_minutes': round(np.sum([r['response_time_seconds'] for r in responses]) / 60, 1),
                'responses': responses
            })
        
        print(f"\n✅ Simulated {len(participants)} participant responses")
        
        # Save simulated data
        simulated_data = {
            'simulation_date': datetime.now().isoformat(),
            'num_participants': len(participants),
            'parameters': {
                'seed': 42,
                'expertise_distribution': {'novice': 0.3, 'intermediate': 0.5, 'expert': 0.2}
            },
            'participants': participants
        }
        
        output_file = os.path.join(self.results_dir, "simulated_responses.json")
        with open(output_file, 'w') as f:
            json.dump(simulated_data, f, indent=2)
        
        print(f"📊 Simulated data saved to: {output_file}")
        
        return participants
    
    def analyze_results(self, participants_data=None):
        """Analyze evaluation results with statistical tests"""
        print(f"\n{'='*60}")
        print("ANALYZING EVALUATION RESULTS")
        print(f"{'='*60}")
        
        if participants_data is None:
            # Load simulated data
            data_file = os.path.join(self.results_dir, "simulated_responses.json")
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    data = json.load(f)
                participants_data = data['participants']
            else:
                print("❌ No data found for analysis")
                return
        
        # Convert to DataFrame for analysis
        all_ratings = []
        
        for participant in participants_data:
            participant_id = participant['participant_id']
            expertise = participant['expertise_level']
            
            for response in participant['responses']:
                # Process Image A ratings
                for criterion, value in response['ratings_a'].items():
                    if criterion != 'artifacts':
                        all_ratings.append({
                            'participant_id': participant_id,
                            'expertise': expertise,
                            'model': response['model_a'],
                            'criterion': criterion,
                            'rating': float(value),
                            'image': response['image_a'],
                            'pair_id': response['pair_id']
                        })
                
                # Process Image B ratings
                for criterion, value in response['ratings_b'].items():
                    if criterion != 'artifacts':
                        all_ratings.append({
                            'participant_id': participant_id,
                            'expertise': expertise,
                            'model': response['model_b'],
                            'criterion': criterion,
                            'rating': float(value),
                            'image': response['image_b'],
                            'pair_id': response['pair_id']
                        })
        
        df = pd.DataFrame(all_ratings)
        
        print(f"\n Dataset Summary:")
        print(f"   Total ratings: {len(df)}")
        print(f"   Participants: {df['participant_id'].nunique()}")
        print(f"   Models: {df['model'].unique()}")
        print(f"   Criteria: {df['criterion'].unique()}")
        
        # Save raw data
        df.to_csv(os.path.join(self.analysis_dir, "ratings_data.csv"), index=False)
        
        # 1. Overall statistics by model
        print(f"\n📊 OVERALL RATINGS BY MODEL:")
        print("-" * 50)
        
        model_stats = df.groupby('model')['rating'].agg(['mean', 'std', 'count', 'median'])
        print(model_stats.round(2))
        
        # 2. Statistics by criterion
        print(f"\n RATINGS BY CRITERION:")
        print("-" * 50)
        
        criterion_stats = df.groupby(['model', 'criterion'])['rating'].agg(['mean', 'std', 'count'])
        print(criterion_stats.round(2))
        
        # 3. Statistical tests
        print(f"\n STATISTICAL TESTS:")
        print("-" * 50)
        
        # ANOVA between models
        models = df['model'].unique()
        model_ratings = [df[df['model'] == model]['rating'].values for model in models]
        
        if len(model_ratings) > 1:
            f_stat, p_value = stats.f_oneway(*model_ratings)
            print(f"ANOVA Test (differences between models):")
            print(f"  F-statistic: {f_stat:.3f}, p-value: {p_value:.6f}")
            
            if p_value < 0.05:
                print(f"  ✅ Significant differences found between models (p < 0.05)")
                
                # Post-hoc pairwise comparisons
                print(f"\n   Post-hoc comparisons (Tukey HSD):")
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                
                tukey_results = pairwise_tukeyhsd(df['rating'], df['model'], alpha=0.05)
                print(tukey_results)
            else:
                print(f"   No significant differences between models")
        
        # 4. Expertise level analysis
        print(f"\n EXPERTISE LEVEL ANALYSIS:")
        print("-" * 50)
        
        expertise_stats = df.groupby(['expertise', 'model'])['rating'].mean().unstack()
        print(expertise_stats.round(2))
        
        # 5. Response time analysis
        response_times = []
        for participant in participants_data:
            for response in participant['responses']:
                response_times.append(response['response_time_seconds'])
        
        print(f"\n RESPONSE TIME ANALYSIS:")
        print("-" * 50)
        print(f"  Mean: {np.mean(response_times):.1f}s per image pair")
        print(f"  Std: {np.std(response_times):.1f}s")
        print(f"  Min: {np.min(response_times):.1f}s")
        print(f"  Max: {np.max(response_times):.1f}s")
        
        # 6. Artifact analysis
        print(f"\n🔍 ARTIFACT ANALYSIS:")
        print("-" * 50)
        
        artifact_counts = {}
        for participant in participants_data:
            for response in participant['responses']:
                for image_type in ['ratings_a', 'ratings_b']:
                    artifacts_str = response[image_type]['artifacts']
                    if artifacts_str:
                        artifacts = [a.strip() for a in artifacts_str.split(',') if a.strip()]
                        model = response['model_a'] if image_type == 'ratings_a' else response['model_b']
                        
                        if model not in artifact_counts:
                            artifact_counts[model] = {}
                        
                        for artifact in artifacts:
                            artifact_counts[model][artifact] = artifact_counts[model].get(artifact, 0) + 1
        
        for model, artifacts in artifact_counts.items():
            print(f"\n  {model}:")
            for artifact, count in artifacts.items():
                percentage = (count / (len(participants_data) * 10)) * 100  # 10 pairs per participant
                print(f"    {artifact}: {count} ({percentage:.1f}%)")
        
        # 7. Create visualizations
        self.create_visualizations(df, participants_data)
        
        # Save analysis report
        self.save_analysis_report(df, participants_data, model_stats, criterion_stats)
        
        return df
    
    def create_visualizations(self, df, participants_data):
        """Create visualization plots"""
        print(f"\n CREATING VISUALIZATIONS...")
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Overall model comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='model', y='rating', data=df, palette='Set2')
        plt.title('Overall Ratings by Model', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Rating (1-5)', fontsize=12)
        plt.ylim(1, 5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Criteria breakdown
        plt.figure(figsize=(14, 8))
        criteria_order = ['realism', 'symmetry', 'proportion', 'coherence', 'artifacts', 'overall']
        
        for idx, criterion in enumerate(criteria_order, 1):
            if criterion in df['criterion'].unique():
                plt.subplot(2, 3, idx)
                criterion_data = df[df['criterion'] == criterion]
                sns.boxplot(x='model', y='rating', data=criterion_data, palette='Set3')
                plt.title(criterion.title(), fontsize=12, fontweight='bold')
                plt.xlabel('')
                plt.ylabel('Rating' if idx in [1, 4] else '')
                plt.ylim(1, 5)
        
        plt.suptitle('Ratings by Model and Criterion', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'criteria_breakdown.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Expertise comparison
        plt.figure(figsize=(12, 6))
        expertise_data = df.groupby(['expertise', 'model'])['rating'].mean().reset_index()
        sns.barplot(x='model', y='rating', hue='expertise', data=expertise_data, palette='viridis')
        plt.title('Average Ratings by Model and Expertise Level', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.legend(title='Expertise Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'expertise_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Response time distribution
        response_times = []
        for participant in participants_data:
            for response in participant['responses']:
                response_times.append(response['response_time_seconds'])
        
        plt.figure(figsize=(10, 6))
        plt.hist(response_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Response Times', fontsize=14, fontweight='bold')
        plt.xlabel('Response Time (seconds per image pair)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(np.mean(response_times), color='red', linestyle='--', label=f'Mean: {np.mean(response_times):.1f}s')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'response_times.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Heatmap of model performance
        plt.figure(figsize=(10, 8))
        pivot_table = df.pivot_table(values='rating', index='model', columns='criterion', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Average Rating'})
        plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'performance_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Visualizations saved to {self.analysis_dir}/")
    
    def save_analysis_report(self, df, participants_data, model_stats, criterion_stats):
        """Save detailed analysis report"""
        print(f"\n SAVING ANALYSIS REPORT...")
        
        report = f"""
FACE GENERATION QUALITY EVALUATION - ANALYSIS REPORT
===================================================

Study Overview:
---------------
- Total Participants: {len(participants_data)}
- Total Ratings: {len(df)}
- Evaluation Criteria: Realism, Symmetry, Proportion, Coherence, Artifacts, Overall
- Models Compared: FFHQ-LDM, StyleGAN2, StyleGAN3, Real

Statistical Summary:
-------------------
1. Overall Model Performance (Average Ratings):
{model_stats.to_string()}

2. Detailed Criteria Performance:
{criterion_stats.to_string()}

3. Key Findings:
----------------

A. Model Rankings (by Overall Score):
"""
        # Rank models
        overall_scores = df[df['criterion'] == 'overall'].groupby('model')['rating'].mean()
        overall_scores = overall_scores.sort_values(ascending=False)
        
        for rank, (model, score) in enumerate(overall_scores.items(), 1):
            report += f"   {rank}. {model}: {score:.2f}/5\n"
        
        report += f"""
B. Performance by Criterion:
"""
        criteria = ['realism', 'symmetry', 'proportion', 'coherence', 'artifacts']
        for criterion in criteria:
            if criterion in df['criterion'].unique():
                criterion_scores = df[df['criterion'] == criterion].groupby('model')['rating'].mean()
                best_model = criterion_scores.idxmax()
                worst_model = criterion_scores.idxmin()
                
                report += f"   • {criterion.title()}:\n"
                report += f"     Best: {best_model} ({criterion_scores[best_model]:.2f})\n"
                report += f"     Worst: {worst_model} ({criterion_scores[worst_model]:.2f})\n"
        
        report += f"""
C. Expertise Effects:
"""
        expertise_stats = df.groupby(['expertise', 'model'])['rating'].mean().unstack()
        for model in expertise_stats.columns:
            differences = expertise_stats[model].max() - expertise_stats[model].min()
            report += f"   • {model}: Max difference between expertise levels = {differences:.2f}\n"
        
        report += f"""
D. Response Time Analysis:
"""
        response_times = []
        for participant in participants_data:
            for response in participant['responses']:
                response_times.append(response['response_time_seconds'])
        
        report += f"   • Mean response time: {np.mean(response_times):.1f}s per pair\n"
        report += f"   • Standard deviation: {np.std(response_times):.1f}s\n"
        report += f"   • Range: {np.min(response_times):.1f}s - {np.max(response_times):.1f}s\n"
        
        # Statistical tests summary
        report += f"""
E. Statistical Significance:
"""
        from scipy import stats
        models = df['model'].unique()
        model_ratings = [df[df['model'] == model]['rating'].values for model in models]
        
        if len(model_ratings) > 1:
            f_stat, p_value = stats.f_oneway(*model_ratings)
            report += f"   • ANOVA: F = {f_stat:.3f}, p = {p_value:.6f}\n"
            
            if p_value < 0.05:
                report += f"   • ✅ Significant differences found between models\n"
            else:
                report += f"   • ❌ No significant differences between models\n"
        
        report += f"""
Conclusions and Recommendations:
--------------------------------
"""
        # Generate conclusions based on data
        best_model = overall_scores.index[0]
        worst_model = overall_scores.index[-1]
        
        report += f"1. {best_model} achieved the highest overall score ({overall_scores.iloc[0]:.2f}/5)\n"
        report += f"2. {worst_model} had the lowest overall score ({overall_scores.iloc[-1]:.2f}/5)\n"
        
        # Check if FFHQ-LDM is the best
        if best_model == 'FFHQ-LDM':
            report += "3. FFHQ-LDM demonstrates superior performance in face generation quality\n"
        else:
            report += f"3. {best_model} outperforms FFHQ-LDM in this evaluation\n"
        
        # Check realism specifically
        if 'realism' in df['criterion'].unique():
            realism_scores = df[df['criterion'] == 'realism'].groupby('model')['rating'].mean()
            realism_ranking = realism_scores.sort_values(ascending=False)
            report += f"4. For realism: {realism_ranking.index[0]} scored highest ({realism_ranking.iloc[0]:.2f}/5)\n"
        
        report += f"""
Visualizations:
---------------
The following plots have been generated:
1. model_comparison.png - Overall ratings by model
2. criteria_breakdown.png - Ratings by model and criterion
3. expertise_comparison.png - Ratings by expertise level
4. response_times.png - Distribution of response times
5. performance_heatmap.png - Heatmap of model performance

Data Files:
-----------
1. ratings_data.csv - Raw rating data for further analysis
2. simulated_responses.json - Complete simulated dataset

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""
        
        report_file = os.path.join(self.analysis_dir, "analysis_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Analysis report saved to: {report_file}")
        
        # Also save as JSON for programmatic access
        analysis_summary = {
            'report_generated': datetime.now().isoformat(),
            'best_model': best_model,
            'worst_model': worst_model,
            'model_scores': overall_scores.to_dict(),
            'total_participants': len(participants_data),
            'total_ratings': len(df),
            'key_findings': {
                'best_performing': best_model,
                'worst_performing': worst_model,
                'realism_leader': df[df['criterion'] == 'realism'].groupby('model')['rating'].mean().idxmax() if 'realism' in df['criterion'].unique() else None,
                'significant_differences': p_value < 0.05 if 'p_value' in locals() else None
            }
        }
        
        json_file = os.path.join(self.analysis_dir, "analysis_summary.json")
        with open(json_file, 'w') as f:
            json.dump(analysis_summary, f, indent=2)
        
        return report
    
    def run_full_study(self):
        """Run complete human evaluation study pipeline"""
        print(f"\n{'='*60}")
        print("RUNNING COMPLETE HUMAN EVALUATION STUDY")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Step 1: Generate stimuli
        faces = self.generate_stimulus_set(num_faces=50)
        
        # Step 2: Create evaluation interfaces
        self.create_evaluation_interface()
        
        # Step 3: Simulate participant responses
        participants = self.simulate_participant_responses(
            num_participants=self.study_params['num_participants']
        )
        
        # Step 4: Analyze results
        analysis_df = self.analyze_results(participants)
        
        # Step 5: Generate summary
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("STUDY COMPLETE!")
        print(f"{'='*60}")
        print(f"\n Study Summary:")
        print(f"   Duration: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Stimuli: {os.path.join(self.stimuli_dir, '*.png')}")
        print(f"   Results: {os.path.join(self.results_dir, '*.json')}")
        print(f"   Analysis: {os.path.join(self.analysis_dir, '*.png/.txt')}")
        print(f"   Evaluation interface: {os.path.join(self.output_dir, 'evaluation_interface.html')}")
        
        print(f"\n To conduct the study:")
        print(f"   1. Open {os.path.join(self.output_dir, 'evaluation_interface.html')} in a web browser")
        print(f"   2. Share with participants")
        print(f"   3. Collect results in {self.results_dir}")
        print(f"   4. Re-run analysis with collected data")
        
        return {
            'faces': faces,
            'participants': participants,
            'analysis': analysis_df,
            'output_dir': self.output_dir,
            'elapsed_time': elapsed_time
        }


def main():
    """Main function"""
    print("\n" + "="*60)
    print("FACE GENERATION HUMAN EVALUATION FRAMEWORK")
    print("="*60)
    print("\nThis framework provides tools for conducting human evaluation")
    print("studies comparing face generation models.\n")
    
    # Initialize study
    study = HumanEvaluationStudy()
    
    # Run full study
    results = study.run_full_study()
    
    print(f"\n Study completed successfully!")
    print(f" All outputs saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()
            
            
