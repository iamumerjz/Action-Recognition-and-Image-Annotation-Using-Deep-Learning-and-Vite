# backend/dl_model/main.py
import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
import pickle
import nltk

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

class Vocabulary:
    """Vocabulary class for image captioning (Flickr8k compatible)"""

    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # Special tokens (ORDER MATTERS!)
        self.pad_token = '<pad>'
        self.start_token = '<start>'
        self.end_token = '<end>'
        self.unk_token = '<unk>'

        for token in [
            self.pad_token,
            self.start_token,
            self.end_token,
            self.unk_token
        ]:
            self.word2idx[token] = self.idx
            self.idx2word[self.idx] = token
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk_token])

    def decode_caption(self, indices, skip_special=True):
        if hasattr(indices, 'cpu'):
            indices = indices.cpu().numpy()

        words = []
        special_ids = {
            self.word2idx[self.pad_token],
            self.word2idx[self.start_token],
            self.word2idx[self.end_token]
        }

        for idx in indices:
            if idx == self.word2idx[self.end_token]:
                break
            if skip_special and idx in special_ids:
                continue
            words.append(self.idx2word.get(idx, self.unk_token))

        return ' '.join(words)


class ActionRecognitionCNN(nn.Module):
    def __init__(self, num_classes=40, dropout=0.5):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights)
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.resnet(x))

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.linear = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        batch_size = features.size(0)
        features = self.adaptive_pool(features)
        features = features.permute(0, 2, 3, 1).view(batch_size, -1, 2048)
        features = self.linear(features)
        features_mean = self.bn(features.mean(dim=1))
        return features, self.dropout(features_mean)

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        return (encoder_out * alpha.unsqueeze(2)).sum(dim=1), alpha

class DecoderLSTM(nn.Module):
    def __init__(self, attention_dim, embed_size, decoder_dim, vocab_size, encoder_dim=256, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        return self.init_h(mean_encoder_out), self.init_c(mean_encoder_out)
    
    def sample_beam_search(self, encoder_out, beam_size=7, max_length=30):
        k = beam_size
        vocab_size = self.vocab_size
        if encoder_out.size(0) > 1:
            encoder_out = encoder_out[0:1]
        encoder_out = encoder_out.expand(k, -1, -1)
        h, c = self.init_hidden_state(encoder_out)
        k_prev_words = torch.LongTensor([[1]] * k).to(encoder_out.device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(encoder_out.device)
        complete_seqs, complete_seqs_scores = [], []
        step = 1
        
        while True:
            embeddings = self.embedding(k_prev_words).squeeze(1)
            attention_weighted, _ = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted = gate * attention_weighted
            h, c = self.decode_step(torch.cat([embeddings, attention_weighted], dim=1), (h, c))
            scores = torch.nn.functional.log_softmax(self.fc(h), dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
            
            prev_word_inds = (top_k_words // vocab_size).long()
            next_word_inds = (top_k_words % vocab_size).long()
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != 2]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            
            k -= len(complete_inds)
            if k == 0 or step > max_length:
                break
            
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            step += 1
        
        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = seqs[0].tolist()
        return torch.LongTensor([seq])

class ImageCaptioningModel(nn.Module):
    def __init__(self, attention_dim, embed_size, decoder_dim, vocab_size, encoder_dim=256, dropout=0.5):
        super().__init__()
        self.encoder = EncoderCNN(encoder_dim)
        self.decoder = DecoderLSTM(attention_dim, embed_size, decoder_dim, vocab_size, encoder_dim, dropout)
    
    def generate_caption(self, image, beam_size=7, max_length=30):
        self.eval()
        with torch.no_grad():
            features, _ = self.encoder(image)
            caption = self.decoder.sample_beam_search(features, beam_size, max_length)
        return caption

class ImageAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_model = None
        self.caption_model = None
        self.vocab = None
        
        self.action_classes = [
            'applauding', 'blowing_bubbles', 'brushing_teeth', 'cleaning_the_floor', 
            'climbing', 'cooking', 'cutting_trees', 'cutting_vegetables', 'drinking', 
            'feeding_a_horse', 'fishing', 'fixing_a_bike', 'fixing_a_car', 'gardening', 
            'holding_an_umbrella', 'jumping', 'looking_through_a_microscope', 
            'looking_through_a_telescope', 'playing_guitar', 'playing_violin', 
            'pouring_liquid', 'pushing_a_cart', 'reading', 'phoning', 'riding_a_bike', 
            'riding_a_horse', 'rowing_a_boat', 'running', 'shooting_an_arrow', 'smoking', 
            'taking_photos', 'texting_message', 'throwing_frisby', 'using_computer', 
            'walking_the_dog', 'washing_dishes', 'watching_TV', 'waving_hands', 
            'writing_on_a_board', 'writing_on_a_book'
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.load_models()
    
    def load_models(self):
        """Load both models"""
        # Load Action Model
        action_model_path = os.path.join(MODELS_DIR, 'best_action_model.pth')
        if os.path.exists(action_model_path):
            try:
                self.action_model = ActionRecognitionCNN(num_classes=len(self.action_classes))
                self.action_model.load_state_dict(torch.load(action_model_path, map_location=self.device))
                self.action_model = self.action_model.to(self.device)
                self.action_model.eval()
                print("Action model loaded")
            except Exception as e:
                print(f"Warning: Could not load action model - {e}")
        else:
            print(f"Warning: Action model not found at: {action_model_path}")
        
        # Load Caption Model
        vocab_path = os.path.join(MODELS_DIR, 'vocabulary.pkl')
        caption_model_path = os.path.join(MODELS_DIR, 'best_caption_model.pth')
        
        if os.path.exists(vocab_path) and os.path.exists(caption_model_path):
            try:
                with open(vocab_path, 'rb') as f:
                    self.vocab = pickle.load(f)
                self.caption_model = ImageCaptioningModel(512, 512, 512, len(self.vocab), 256, 0.5)
                self.caption_model.load_state_dict(torch.load(caption_model_path, map_location=self.device))
                self.caption_model = self.caption_model.to(self.device)
                self.caption_model.eval()
                print("Caption model loaded")
            except Exception as e:
                print(f"Warning: Could not load caption model - {e}")
        else:
            print(f"Warning: Caption model files not found")
            print(f"  Vocab: {vocab_path}")
            print(f"  Model: {caption_model_path}")
    
    def predict_action(self, image_tensor):
        """Predict action from image"""
        if not self.action_model:
            return None, 0.0
        
        with torch.no_grad():
            outputs = self.action_model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        return self.action_classes[pred.item()], conf.item()
    
    def predict_caption(self, image_tensor):
        """Generate caption for image"""
        if not self.caption_model or not self.vocab:
            return None
        
        generated = self.caption_model.generate_caption(image_tensor, beam_size=7, max_length=30)
        return self.vocab.decode_caption(generated[0], skip_special=True)
    
    def analyze_image(self, image_path):
        """Analyze image and return action + caption"""
        if not os.path.exists(image_path):
            return {
                'error': f"Image not found: {image_path}",
                'action': None,
                'confidence': 0.0,
                'caption': None
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            action, confidence = self.predict_action(image_tensor)
            caption = self.predict_caption(image_tensor)
            
            return {
                'error': None,
                'action': action,
                'confidence': confidence,
                'caption': caption
            }
        
        except Exception as e:
            return {
                'error': f"Error processing image: {str(e)}",
                'action': None,
                'confidence': 0.0,
                'caption': None
            }

def main():
    """Main CLI function"""
    print("="*70)
    print(" IMAGE ANALYZER - Action Recognition + Captioning")
    print("="*70)
    print()
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize analyzer
    print("Loading models...")
    analyzer = ImageAnalyzer()
    print()
    
    # Check which models are loaded
    models_loaded = []
    if analyzer.action_model:
        models_loaded.append("Action Recognition")
    if analyzer.caption_model:
        models_loaded.append("Image Captioning")
    
    if not models_loaded:
        print("ERROR: No models loaded!")
        print()
        print("Please ensure the following files exist:")
        print(f"  - {os.path.join(MODELS_DIR, 'best_action_model.pth')} (for action recognition)")
        print(f"  - {os.path.join(MODELS_DIR, 'best_caption_model.pth')} + {os.path.join(MODELS_DIR, 'vocabulary.pkl')} (for captioning)")
        print()
        sys.exit(1)
    
    print(f"Models loaded: {', '.join(models_loaded)}")
    print()
    
    # Analyze image
    print(f"Analyzing: {image_path}")
    print("-"*70)
    
    result = analyzer.analyze_image(image_path)
    
    if result['error']:
        print(f"ERROR: {result['error']}")
        sys.exit(1)
    
    # Display results
    print()
    print("="*70)
    print(" RESULTS")
    print("="*70)
    print()
    
    if result['action']:
        action_formatted = result['action'].replace('_', ' ').title()
        print(f"ACTION: {action_formatted}")
        print(f"Confidence: {result['confidence']:.2%}")
        print()
    
    if result['caption']:
        print(f"CAPTION: {result['caption']}")
        print()
    
    print("="*70)

if __name__ == "__main__":
    main()