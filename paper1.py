import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse

class PredictiveModel(nn.Module):
    """
    Convolutional neural network for pixel prediction as described in the paper.
    Uses a structure similar to the paper's architecture with multiple conv layers.
    """
    def __init__(self):
        super(PredictiveModel, self).__init__()
        # Feature extraction and prediction network
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.layers(x)

class StegoDataset(Dataset):
    """Dataset for training the predictive model"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) 
                           if os.path.isfile(os.path.join(root_dir, f)) and 
                           f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # The model is trained to predict the original image
        return image, image

class ReversibleSteganography:
    """Main class implementing the reversible steganography system"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = PredictiveModel().to(device)
        self.location_map = None
    
    def train(self, train_loader, val_loader=None, epochs=10, lr=0.001):
        """Train the predictive model"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}')
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.6f}')
    
    def save_model(self, path):
        """Save the model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load the model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
    
    def predict_image(self, image):
        """Generate prediction for an image"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                # Convert numpy array to tensor
                transform = transforms.ToTensor()
                image_tensor = transform(Image.fromarray(image.astype(np.uint8))).unsqueeze(0)
            else:
                image_tensor = image.unsqueeze(0)
            
            image_tensor = image_tensor.to(self.device)
            prediction = self.model(image_tensor)
            
        return prediction.squeeze(0).cpu()
    
    def calculate_prediction_errors(self, original, prediction):
        """Calculate prediction errors as described in the paper"""
        # Convert tensors to numpy if needed
        if isinstance(original, torch.Tensor):
            original = original.permute(1, 2, 0).numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.permute(1, 2, 0).numpy()
        
        # Prediction error calculation
        pe = original - prediction
        return pe
    
    def embed(self, cover_image, message_bits):
        """
        Embed message bits using reversible steganography
        
        Args:
            cover_image: The cover image (numpy array or tensor)
            message_bits: Binary message to embed (string of 0s and 1s)
            
        Returns:
            stego_image: The image with hidden data
            location_map: Map of modified pixels for extraction
        """
        # Convert cover image to tensor if it's a numpy array
        if isinstance(cover_image, np.ndarray):
            transform = transforms.ToTensor()
            cover_tensor = transform(Image.fromarray(cover_image.astype(np.uint8)))
        else:
            cover_tensor = cover_image.clone()
        
        # Get prediction
        prediction = self.predict_image(cover_tensor)
        
        # Calculate prediction errors
        cover_np = cover_tensor.permute(1, 2, 0).numpy() * 255.0
        pred_np = prediction.permute(1, 2, 0).numpy() * 255.0
        pe = self.calculate_prediction_errors(cover_np, pred_np)
        
        # Create location map for pixel selection (as per the paper's methodology)
        # Sort pixels by prediction error magnitude
        h, w, c = pe.shape
        pixel_errors = []
        for i in range(h):
            for j in range(w):
                error_val = np.mean(np.abs(pe[i, j, :]))
                pixel_errors.append((error_val, i, j))
        
        # Sort pixels by prediction error (smaller errors are better for embedding)
        pixel_errors.sort(key=lambda x: x[0])
        
        # Embed message bits in the best locations
        stego_image = cover_np.copy()
        location_map = {}
        message_index = 0
        
        for _, i, j in pixel_errors:
            if message_index >= len(message_bits):
                break
            
            # Implementation of the histogram shifting method
            # For simplicity, we'll modify the LSB of the blue channel
            bit = int(message_bits[message_index])
            
            # Record the original pixel value and position
            location_map[message_index] = (i, j, stego_image[i, j, 0])
            
            # Modify the pixel based on the embedding bit
            if bit == 0:
                # Shift down
                stego_image[i, j, 0] = stego_image[i, j, 0] - 1 if stego_image[i, j, 0] > 0 else 0
            else:
                # Shift up
                stego_image[i, j, 0] = stego_image[i, j, 0] + 1 if stego_image[i, j, 0] < 255 else 255
            
            message_index += 1
        
        # Save location map for extraction
        self.location_map = location_map
        
        return stego_image.astype(np.uint8), location_map
    
    def extract(self, stego_image, location_map, message_length):
        """
        Extract message bits from stego image
        
        Args:
            stego_image: The stego image with hidden data
            location_map: Map of modified pixels
            message_length: Length of the hidden message in bits
            
        Returns:
            message_bits: Extracted binary message
            recovered_image: The recovered cover image
        """
        # Make a copy of stego image for recovery
        recovered_image = stego_image.copy()
        
        # Extract the message bits
        message_bits = ''
        
        for i in range(message_length):
            if i not in location_map:
                break
                
            pos_i, pos_j, original_value = location_map[i]
            current_value = stego_image[pos_i, pos_j, 0]
            
            # Compare with original to determine bit
            if current_value < original_value:
                message_bits += '0'
            else:
                message_bits += '1'
            
            # Restore the original pixel value
            recovered_image[pos_i, pos_j, 0] = original_value
        
        return message_bits, recovered_image
    
    def binary_to_text(self, binary_string):
        """Convert binary string to text"""
        text = ''
        for i in range(0, len(binary_string), 8):
            byte = binary_string[i:i+8]
            if len(byte) == 8:
                text += chr(int(byte, 2))
        return text
    
    def text_to_binary(self, text):
        """Convert text to binary string"""
        binary = ''
        for char in text:
            binary += format(ord(char), '08b')
        return binary
    
    def evaluate(self, original_image, stego_image):
        """Calculate PSNR and other metrics between original and stego images"""
        mse = np.mean((original_image - stego_image) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Calculate embedding capacity
        h, w, _ = original_image.shape
        capacity = h * w  # Theoretical maximum in bits
        
        return {
            'PSNR': psnr,
            'MSE': mse,
            'Capacity': capacity
        }

# Main function to run the implementation
def main():
    parser = argparse.ArgumentParser(description='Reversible Steganography with Deep Learning')
    parser.add_argument('--train', action='store_true', help='Train the predictive model')
    parser.add_argument('--embed', action='store_true', help='Embed a message in an image')
    parser.add_argument('--extract', action='store_true', help='Extract a message from an image')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory with training images')
    parser.add_argument('--model_path', type=str, default='./model.pth', help='Path to save/load model')
    parser.add_argument('--cover_path', type=str, help='Path to cover image')
    parser.add_argument('--stego_path', type=str, default='./stego.png', help='Path to save/load stego image')
    parser.add_argument('--message', type=str, default='Secret message', help='Message to hide')
    args = parser.parse_args()
    
    stego_system = ReversibleSteganography()
    
    if args.train:
        # Set up data transformations
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Create dataset and data loaders
        train_dataset = StegoDataset(args.data_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Train the model
        stego_system.train(train_loader, epochs=5)
        stego_system.save_model(args.model_path)
        print(f"Model saved to {args.model_path}")
    
    if args.embed:
        # Load the trained model
        stego_system.load_model(args.model_path)
        
        # Load cover image
        cover_image = np.array(Image.open(args.cover_path).convert('RGB'))
        
        # Convert message to binary
        binary_message = stego_system.text_to_binary(args.message)
        print(f"Message to hide: {args.message}")
        print(f"Binary message length: {len(binary_message)} bits")
        
        # Embed the message
        stego_image, location_map = stego_system.embed(cover_image, binary_message)
        
        # Save the stego image
        Image.fromarray(stego_image).save(args.stego_path)
        print(f"Stego image saved to {args.stego_path}")
        
        # Evaluate the result
        metrics = stego_system.evaluate(cover_image, stego_image)
        print(f"PSNR: {metrics['PSNR']:.2f} dB")
        print(f"MSE: {metrics['MSE']:.4f}")
        print(f"Theoretical max capacity: {metrics['Capacity']} bits")
        
        # Visualize the results
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cover_image)
        plt.title('Original Cover Image')
        plt.subplot(1, 2, 2)
        plt.imshow(stego_image)
        plt.title('Stego Image')
        plt.tight_layout()
        plt.savefig('comparison.png')
        plt.show()
        
    if args.extract:
        # Load the trained model
        stego_system.load_model(args.model_path)
        
        # Load stego image
        stego_image = np.array(Image.open(args.stego_path).convert('RGB'))
        
        # For extraction, we need the location map that was created during embedding
        # In a real system, this would be embedded in the image or transmitted separately
        if stego_system.location_map is None:
            print("Error: Location map not available. Run embedding first.")
            return
        
        # Calculate the binary message length from the original message
        binary_length = len(stego_system.text_to_binary(args.message))
        
        # Extract the message
        extracted_binary, recovered_image = stego_system.extract(
            stego_image, stego_system.location_map, binary_length)
        
        # Convert binary to text
        extracted_message = stego_system.binary_to_text(extracted_binary)
        
        print(f"Extracted message: {extracted_message}")
        print(f"Original message: {args.message}")
        
        # Verify the recovered image
        original_cover = np.array(Image.open(args.cover_path).convert('RGB'))
        recovery_mse = np.mean((original_cover - recovered_image) ** 2)
        if recovery_mse < 1e-10:
            print("Cover image successfully recovered!")
        else:
            print(f"Recovery MSE: {recovery_mse:.10f}")

if __name__ == "__main__":
    main()