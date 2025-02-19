import streamlit as st
from PIL import Image
import torch
import torchvision.models as models  # Add this line
import torchvision.transforms as transforms


# Load the saved model
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)  # 4 classes
    model.load_state_dict(torch.load('brain_tumor_resnet50.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Predict the class
def predict(image, model):
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return preds.item()

# Class labels
class_labels = ['glioma', 'healthy', 'meningioma', 'pituitary']

# Streamlit app
def main():
    st.title("Brain Tumor Classification using ResNet-50")
    st.write("Upload an MRI image to classify the type of brain tumor.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', use_container_width=True)
        
        # Preprocess and predict
        st.write("Classifying...")
        model = load_model()
        image_tensor = preprocess_image(image)
        prediction = predict(image_tensor, model)
        
        # Display the result
        st.success(f"Prediction: {class_labels[prediction]}")

if __name__ == "__main__":
    main()
