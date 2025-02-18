# Novel-AI-based-ensemble-technique-for-efficient-training-of-DNN-on-Brain-Tumor-Detection


# Abstract:

Identifying brain tumors in MRI images is an essential medical imaging endeavor that helps with early diagnosis &amp; therapy. Current solutions result in poor accuracy &amp; misclassification. To categorize brain tumors into 4 groups gliomas, meningiomas, pituitary tumors, healthy cases. Our proposed approach was CNNs &amp; ResNet-50 with accuracy of 98.19%. 


# I. INTRODUCTION

Brain tumors are a neurological condition that require prompt, accurate, and efficient diagnosis for effective treatment. Reduced survival rates and poor prognoses can result from delayed diagnosis. Because magnetic resonance imaging (MRI) can produce precise, high-resolution, and contrast-rich images of soft tissues, it is an ideal imaging technique for identifying brain tumors. However, manual MRI scan analysis takes a lot of time, is prone to mistakes, and calls for specific knowledge from neurologists and radiologists.

# II. LITERATURE REVIEW 

Deep learning techniques, particularly Convolutional Neural Networks (CNNs) and Residual Networks (ResNet-50), have demonstrated significant potential for automated tumor detection and classification. The integration of CNNs and ResNet-50 in an ensemble learning framework has emerged as an advanced technique to improve the accuracy and robustness of brain tumor detection. This literature review explores existing studies on CNNs, ResNet-50, ensemble learning, and their applications in brain tumor classification using MRI images.


Early approaches to brain tumor detection relied on manual feature extraction and classical machine learning models such as Support Vector Machines (SVMs), K-Nearest Neighbors (KNN), Random Forest (RF), and Decision Trees (DTs). These methods require handcrafted feature extraction techniques such as wavelet transform, histogram analysis, and texture features to identify tumor characteristics. Several studies have demonstrated CNN-based approaches for brain tumor detection.


Hossain et al. (2021) developed a custom CNN model for classifying brain tumors into glioma, meningioma, and pituitary tumors, achieving an accuracy of 92%. However, the model struggles with data generalization. Islam et al. (2020) implemented a deep CNN model trained on augmented MRI datasets, improving classification performance but requiring large computational resources. Sajjad et al. (2019) proposed a hybrid CNN model combined with transfer learning techniques to enhance the tumor classification accuracy to 94%.

Gupta et al. (2022) used ResNet-50 with transfer learning to classify brain tumors in MRI images, achieving an accuracy of 96.2%. Sharma et al. (2021) proposed a ResNet-50 model combined with CNN, demonstrating superior performance over standalone CNN models with an accuracy of 97%. Patel et al. (2020) fine-tuned ResNet-50 on a large dataset of brain tumor MRI scans, improving the generalization across different MRI machines and scanning conditions. Despite its strengths, ResNet-50 alone may struggle with overfitting small medical datasets. To further enhance accuracy, ensemble learning techniques combining CNN and ResNet-50 have been explored.

Khan et al. (2022) developed an ensemble learning framework combining a CNN and ResNet-50, achieving an accuracy of 98% on the BraTS MRI dataset. Rajesh et al. (2021) proposed a hybrid CNN-ResNet model that outperformed individual CNN and ResNet models, reducing false-positive rates in tumor detection. Ali et al. (2020) experimented with different ensemble techniques (hard voting, soft voting, weighted averaging) and found that a CNN-ResNet ensemble performed best in MRI-based tumor classification. It achieves up to 98% accuracy and improves early tumor detection. More reliable predictions, reducing false positives and negatives. Can analyze MRI scans in real time, helping doctors make quicker decisions. It can be deployed in hospitals, telemedicine platforms, and AI-assisted diagnostics.


# III. METHODOLOGY

# A) Dataset:

We compiled a comprehensive dataset of brain tumor images to facilitate the classification of normal and tumor-affected brain tissues. The images were sourced from an open-access repository on the Kaggle platform to ensure the availability of diverse, high-quality data for our research. In total, our dataset consisted of 7,023 MRI scans, categorized into 5,023 tumor-affected images and 2,000 images from healthy individuals. The inclusion of a substantial number of normal brain scans is crucial for developing a robust classification model that can accurately distinguish between pathological and nonpathological cases. Examples of tumors and healthy brain tissues are displayed.
![Screenshot_20250202_144825_Chrome](https://github.com/user-attachments/assets/c0e8bd89-e079-4906-ad5e-6074c366a2ba)


# B) Proposed Model:

This proposed model aims to classify brain MRI scans into four categories: glioma, healthy, meningioma, and pituitary tumors. The architecture is based on ResNet-50, a deep convolutional neural network (CNN) pre-trained on ImageNet, which has been fine-tuned for the specific task of brain tumor classification. the proposed ResNet-50 model aims to achieve high accuracy and reliability in classifying brain MRI scans. The model's ability to distinguish between glioma, meningioma, pituitary tumors, and healthy brain scans will aid in early diagnosis and treatment planning, contributing to improved healthcare outcomes.


# Data Preparation:

The demonstrate utilizes an MRI brain check dataset. The dataset is organized into distinctive envelopes, each comparing to a specific lesson. The dataset is part into 70% of Preparing Information, 15% of Approval Information, 15% of Test Information. A subclass of datasets.ImageFolder is utilized for stacking pictures. Albumentations-based changes are connected for expansion.


# Data Augmentation:

To improve generalization and reduce overfitting, Albumentations-based augmentation techniques are applied. The training set augmentations include resizing all images to 224x224 to ensure uniform size, randomly cropping images to 224x224, applying horizontal flipping, adjusting brightness and contrast randomly, rotating images within a range of ±15 degrees, randomly removing small regions using coarse dropout, and normalizing pixel values for better training convergence. For the validation and test sets, only resizing to 224x224 and normalization transformations are applied to maintain consistency and ensure reliable model evaluation.


# Model Architecture:

The deep learning model is built upon ResNet-50, a 50-layer deep CNN known for its superior feature extraction capabilities. The model is initialized with ImageNet-pre-trained weights to transfer knowledge from a large-scale dataset. The final fully connected layer is replaced to classify images into four classes instead of the original 1000 ImageNet classes, and the output layer consists of a fully connected layer with four neurons activated by Softmax.


# Training Process:

The training follows a structured process where CrossEntropyLoss is used as the loss function since it is suitable for multi-class classification problems. The Adam optimizer is employed with a learning rate of 0.001 to enable adaptive learning and prevent vanishing gradients. A learning rate scheduler, ReduceLROnPlateau, monitors validation loss and reduces the learning rate by a factor of 0.1 if the loss does not improve for five epochs. The training strategy includes a batch size of 32 and a total of 25 epochs. The training loop consists of a forward pass where the input is passed through the network, computation of loss using cross-entropy loss, backpropagation to compute gradients and update weights, and evaluation of the model on the validation set.


# Model Evaluation:

The performance metrics indicate that the validation accuracy reached 97.43% while the test accuracy achieved 98.19%. The confusion matrix and classification report demonstrate high precision and recall across all classes. The training progress is visualized through the training and validation loss curve along with the validation accuracy curve, providing insights into model performance and convergence.


# CNN Algorithm:

The Convolutional Neural Network (CNN) algorithm used in ResNet-50 is designed for deep feature extraction and efficient image classification. It consists of multiple convolutional layers that automatically learn spatial hierarchies of features, capturing essential patterns in MRI scans. ResNet-50 incorporates residual learning, enabling deeper networks by introducing shortcut connections that help mitigate vanishing gradient issues and enhance training stability. The model processes input images through convolutional, batch normalization, activation, and pooling layers before passing them to fully connected layers for final classification. With its ability to capture complex patterns and transfer learning from large-scale datasets, ResNet-50 proves to be highly effective in brain tumor detection, achieving superior accuracy and robustness in medical imaging applications.
![Screenshot_20250218_150506_WhatsApp](https://github.com/user-attachments/assets/140526fb-b6e5-4dcb-8235-c52c984a5412)
![Screenshot_20250218_150516_WhatsApp](https://github.com/user-attachments/assets/79088522-83dd-4ec2-98e4-cf86cbf28dbb)


ResNet-50 is a deep convolutional neural network (CNN) model, part of the ResNet (Residual Networks) family, introduced by Microsoft Research in the paper "Deep Residual Learning for Image Recognition" in 2015. The ResNet-50 variant has 50 layers and was designed to solve the problem of vanishing gradients in very deep networks by introducing residual blocks. These blocks allow the network to learn residual functions instead of directly mapping inputs to outputs, making it easier for the network to learn identity functions and improve convergence.


# ResNet-50 Model:

ResNet's hallmark feature is the residual block, which contains skip connections that bypass one or more layers, helping to combat the vanishing gradient problem in deep networks. Specifically, ResNet-50 has 50 layers, though there are other variants with more layers such as ResNet-101 and ResNet-152. The model consists of convolutional layers, batch normalization, activation functions, and pooling operations, with the key building block being the residual unit, which includes two 3x3 convolutions with a skip connection. ResNet-50 has demonstrated impressive performance in various image classification tasks, including those on large datasets like ImageNet, and is known for being both highly accurate and computationally efficient.
![Screenshot_20250218_150955_WhatsApp](https://github.com/user-attachments/assets/6e6dae97-2399-414b-9456-f8e53c632174)


# IV. EVALUATION METRICS

# Loss Function (Criterion):

The loss function used here is CrossEntropyLoss, which is commonly used for multi-class classification tasks. Cross-entropy loss measures how different the predicted probability distribution is from the actual labels.


# Accuracy Calculation:

Accuracy is a metric that measures how many predictions the model got correct. Accuracy is a good metric when the dataset is balanced (i.e., all classes have a similar number of samples). However, for imbalanced datasets, accuracy alone may not be sufficient.
accuracy = (no of correct predictions/ total no of predictions) × 100


# Evaluation during training:

After each epoch, the model is evaluated on the validation set. If validation loss increases while training loss decreases, it indicates overfitting. If both training and validation loss are high, it indicates underfitting. Validation accuracy tells how well the model generalizes to unseen data.


# Learning Rate Adjustment Using Scheduler:

The ReduceLROnPlateau scheduler reduces the learning rate when validation loss does not improve for patience epochs. This prevents the model from getting stuck in a local minimum.


# Final Evaluation on the Test Set:

After training, the model is evaluated on the test dataset. The test loss and test accuracy indicate how well the model performs on completely unseen data.


# Example of the final results:

# Test Loss (0.07): 

A low test loss suggests that the model makes very few errors on the test set. Test Accuracy (98.2%): A high accuracy means the model correctly classifies most test samples.
![Screenshot_20250202_144854_Chrome](https://github.com/user-attachments/assets/5375ed07-a8ee-4b39-8134-01f51e318075)
![Screenshot_20250202_144858_Chrome](https://github.com/user-attachments/assets/ace5a9bc-7d3a-40f7-9b16-355d372ee2de)
![Screenshot_20250202_144908_Chrome](https://github.com/user-attachments/assets/da48a60e-a8bc-4090-9828-93a9ce015eda)


# V. CONCLUSION

The ResNet-50 model demonstrates excellent performance in classifying brain tumor types with an overall accuracy of 98 percent on the test dataset. The classification report indicates that the model performs exceptionally well across all four categories with precision recall and F1-scores close to 1.00 highlighting its reliability. Glioma and meningioma classes show slightly lower recall values suggesting minor misclassifications but overall the model generalizes well. The high precision and recall values in the healthy and pituitary tumor classes indicate that the model correctly identifies these cases with minimal false positives and false negatives. The macro and weighted averages also confirm the balanced performance across all classes making ResNet-50 a highly effective choice for brain tumor classification.


# REFERENCES

[1] M. Rizwan, A. Shabbir, A. R. Javed, M. Shabbir, T. Baker and D. Al-Jumeily Obe, “Brain Tumor and Glioma Grade Classification Using Gaussian Convolutional Neural Network,” in IEEE Access, vol. 10, pp. 29731-29740, 2022, doi: 10.1109/ ACCESS.2022.3153108.
[2] H. H. Sultan, N. M. Salem and W. Al-Atabany, “Multi-Classification of Brain Tumor Images Using Deep Neural Network,” in IEEE Access, vol. 7, pp. 69215-69225, 2019, doi: 10.1109/ACCESS.2019.2919122.
[3] D. J. Hemanth, J. Anitha, A. Naaji, O. Geman, D. E. Popescu and L. Hoang Son, “A Modified Deep Convolutional Network for Abnormal Brain Image Classification,” in IEEE Access, vol. 7, pp. 4275-4283, 2019, doi: 10.1109/ACCESS.2018.2885639.
[4] S. Das, O. F. M. R. R. Aranya and N. N. Labiba, “Brain Tumor Classification Using Convolutional Neural Network,” 2019 1st International Conference on Advances in Science, Engineering and Robotics Technology (ICASERT), Dhaka, Bangladesh, 2019, pp. 1-5, doi: 10.1109/ICASERT.2019.b6.
[5] S. M. Fayadh, “Automatic Brain Cancer Recognition in CT-Scan Images,” 2022, pp. 76-81, doi: 10.1109/IICCIT55816.2022.b7.
[6] A. S. Musallam, A. S. Sherif and M. K. Hussein, “A New Convolutional Neural Network Architecture for Automatic Detection of Brain Tumors in Magnetic Resonance Imaging Images,” in IEEE Access, vol. 10, pp. 2775-2782, 2022, doi: 10.1109/ACCESS.2022.3140289.
[7] A. Vidyarthi, R. Agarwal, D. Gupta, R. Sharma, D. Draheim and P. Tiwari, “Machine Learning Assisted Methodology for Multiclass Classification of Malignant Brain Tumors,” in IEEE Access, vol. 10, pp. 50624-50640, 2022, doi: 10.1109/ACCESS.2022.3172303.
[8] S. Pereira, A. Pinto, V. Alves and C. A. Silva, “Brain Tumor Segmentation Using Convolutional Neural Networks in MRI Images,” in IEEE Transactions on Medical Imaging, vol. 35, no. 5, pp. 1240-1251, May 2016, doi: 10.1109/TMl.2016.2538465.
[9] S. Solanki, U. P. Singh, S. S. Chouhan and S. Jain, “Brain Tumor Detection and Classification Using Intelligence Techniques: An Overview,” in IEEE Access, vol. 11, pp. 12870-12886, 2023, doi: 10.1109/ACCESS.2023.3242666.
[10] N. M. Dipu, S. A. Shohan and K. M. A. Salam, “Deep Learning Based Brain Tumor Detection and Classification,” 2021 International Conference on Intelligent Technologies (CONIT), Hubli, India, 2021, pp. 1-6, doi: 10.1109/CONIT51480.2021.b10.
[11] A. A. Dehkordi, M. Hashemi, M. Neshat, A. Mirjalili, and A. S. Sadiq, “A Novel Evolutionary Convolutional Neural Network for the Identification and Categorization of Brain Tumors,” SSRN Electron.J., 2022, doi: 10.2139/ssrn.4292650..
[12] Md. Saikat Islam Khan., Anichur Rahman., Tanoy Debnath., Md. Razaul Karim, Mostofa Kamal Nasir, Shahab S Band, Amir Mosavi & Iman Dehzangi (2022). Efficient identification of brain tumors by deep convolutional neural networks, doi: 10.1016/j.csbj.2022.08.039.
[13] Abdullah A. Asiri, Ahmad Shaf, Tariq Ali, Md. Aamir, Md. Irfan, Saeed Alqahtani, Khlood M. Mehdar, Hanan Talal Halawani, Ali H. Alghamdi, Abdullah Fahad A. Alshamrani & Samar M. Alqhtani (2023). Brain Tumor Identification and Categorization using Optimized CNN with ResNet50 and U-Net Model: An Investigation on TCGA-LGG and TCIA Dataset for MRI Uses, doi: 10.3390/life 13071449.
[14] Shtwai Alsubai., Habib Ullah Khan., Abdullah Alqahtani., Md. Sha, Sidra Abbas, Uzma Ghulam Mohammad, Group Deep Learning for identifying brain tumors, doi: 10.3389/fncom.2022.1005617.
[15] Kadry S., Nam Y., Rajinikanth V., Rauf H.T., Lawal I.A. Proceedings of the Seventh International Conference on Bio Signals, Images, and Instrumentation (ICBSII), Chennai, India, 2021; Automated Detection of Brain Abnormality using Deep-Learning-Scheme: A Study. March 25–27, 2021
