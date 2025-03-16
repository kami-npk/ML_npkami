import streamlit as st
# ตั้งค่าธีมสีเข้ม
st.set_page_config(page_title="AI Overview", layout="wide", initial_sidebar_state="expanded")
import numpy as np
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf  # ใช้ TensorFlow โหลดโมเดล
import keras
from tensorflow.python.keras.models import load_model
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
import streamlit as st
import torchvision.transforms as transforms
import pandas as pd

# โหลดข้อมูล
df = pd.read_csv("dataset/csgo_round_snapshots.csv")
# ✅ ตรวจสอบ DataFrame
if df.empty:
    st.error("❌ Error: DataFrame is empty! Please check your CSV file.")
    st.stop()

@st.cache_resource
def load_models():
    rf_model = joblib.load("models/rf1_model.pkl")
    dnn_model = keras.models.load_model("models/dnn_modelComplete.h5")
    return rf_model, dnn_model
rf_model, dnn_model = load_models()

# ✅ ฟังก์ชันทำ Prediction
def predict_winner(model, features):
    features = np.array(features).reshape(1, -1)  
    prediction = model.predict(features)

    if isinstance(model, tf.keras.Model):  
        predicted_team = "T" if prediction[0][0] > 0.5 else "CT"
        print(f"Raw Prediction Output: {prediction}, Predicted Team: {predicted_team}")  # ✅ Debug
        return predicted_team
    else:  
        return "T" if prediction[0] == 1 else "CT"

# โหลดโมเดล ResNet18 และแก้ไขชั้น Fully Connected
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 11)  # เปลี่ยนเลข 5 เป็นจำนวนคลาสที่เทรน
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # ตั้งค่าเป็นโหมดประเมินผล
    return model

# โหลดโมเดล
model = load_model("models/csgo_weapon_classifier.pth")
# แปลงภาพให้เป็น Tensor
def transform_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")  # แปลงให้เป็น RGB
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ปรับขนาด
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # เพิ่ม batch dimension

# พยากรณ์คลาสของอาวุธจากภาพ
def predict(image):
    image = transform_image(image)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()  # คืนค่าหมายเลขคลาสที่พยากรณ์ได้

# สไตล์ CSS เพิ่มเติม
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #252526;
        }
    </style>
""", unsafe_allow_html=True)

# สร้าง Sidebar
st.sidebar.title("CS:GO Machine Learning & Neural Network")
page = st.sidebar.radio("", [ "Machine Learning Detail", "Machine Learning Model", "Neural Network Detail", "Neural Network Model"])

# เนื้อหาแต่ละหน้า    
if page == "Machine Learning Detail":
    
    col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

    with col2:
        st.image("csgo.png", caption="Counter-Strike: Global Offensive", width=600)  # Adjust width
    with col1:
        st.title("Machine Learning Detail")
        st.markdown("""
            ## CS:GO Introduction""")
    
    st.markdown("""

CS:GO is a tactical shooter, where two teams (CT and Terrorist) play for a best of 30 rounds, with each round being 1 minute and 55 seconds. There are 5 players on each team (10 in total) and the first team to reach 16 rounds wins the game. At the start, one team plays as CT and the other as Terrorist. After 15 rounds played, the teams swap side. There are 7 different maps a game can be played on. You win a round as Terrorist by either planting the bomb and making sure it explodes, or by eliminating the other team. You win a round as CT by either eliminating the other team, or by disarming the bomb, should it have been planted.


### The total number of snapshots is 122411. 
This website uses a Keres DNN to predict round winners, either Counter-Terrorist or Terrorist.

Key takeways:
Keras can be used to make DNN's that fit the problem well.
Model tuning requires a lot of experimentation.

---

##  Dataset: CS:GO Round Winner Classification (Kaggle)
The dataset used for training these models comes from **[CS:GO Round Winner Classification](https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification)** on Kaggle. This dataset contains a large number of recorded rounds from competitive CS:GO matches.

**Dataset Details:**
- Contains thousands of CS:GO rounds with multiple game-related features.
- Collected from real competitive matches, providing realistic gameplay scenarios.
- Used to train models to predict whether the **Counter-Terrorists (CT)** or **Terrorists (T)** will win the round.

**Key Features in the Dataset:**
| Feature        | Description |
|---------------|------------|
| **time_left**  | Time remaining in the round |
| **ct_score**   | Counter-Terrorist team score |
| **t_score**    | Terrorist team score |
| **map**        | The current map being played |
| **bomb_planted** | Whether the bomb is planted (True/False) |
| **ct_health**  | Total health of the CT team |
| **t_health**   | Total health of the T team |
| **ct_armor**   | Total armor of the CT team |
| **t_armor**    | Total armor of the T team |

---

## 1️ Deep Neural Network (DNN)
A **Deep Neural Network (DNN)** is a type of Artificial Neural Network that consists of multiple layers. In this project, the DNN model is trained using **Keras & TensorFlow**. It takes several inputs such as team scores, health, armor, bomb status, and map information to predict the round winner.

**Key Features:**
- Uses multiple fully connected layers with activation functions.
- Optimized using Adam optimizer and categorical crossentropy loss.
- Learns complex patterns from CS:GO round data.

---

## 2️ Random Forest Classifier
The **Random Forest** model is an ensemble learning method that combines multiple decision trees to improve accuracy. Unlike the DNN, it does not require extensive training and hyperparameter tuning.

**Key Features:**
- Based on Decision Trees, making it easier to interpret.
- Less prone to overfitting compared to a single Decision Tree.
- Works well even with smaller datasets.

---

##  Which Model is Better?
- **DNN** is more powerful and can capture complex interactions in data but requires more training time.
- **Random Forest** is faster, easier to train, and can work well even with smaller datasets.

Both models provide valuable insights into predicting CS:GO round winners. You can switch between them in the **"ML Model"** page.

---
""")
    st.markdown("""
## Before training the model How to preprocess the data?
### Data Preparation and Training Process for the DNN Model
1. **Encoding Categorical Variables**  
   - Columns such as `map` and `bomb_planted` contain categorical values.  
   - We use **One-Hot Encoding (OHE)** to convert these into numerical format.  

   ```python
   object_cols = ['map', 'bomb_planted']
   ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
   X_encoded = pd.DataFrame(ohe.fit_transform(X[object_cols]))
   X_encoded.columns = ohe.get_feature_names_out(object_cols)
   X_encoded.index = X.index
   X = pd.concat([X.drop(object_cols, axis=1), X_encoded], axis=1
   ```
    """)
    
    st.markdown("""

2. **Encoding Target Variable**  
   - The target variable winner (either CT or T) is encoded into numerical labels using Label Encoding.
   - "CT" is mapped to 0, and "T" is mapped to 1.

    ```python
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # "CT" → 0, "T" → 1
    ```
    """)
    st.markdown("""
3. **Feature Scaling**  
   - Since features have different scales, we apply Standard Scaling to normalize them.
   - This ensures that the model learns efficiently without certain features dominating others.

   ```python
   scaler = StandardScaler()
   X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
   ```
    """)
    

    st.markdown("""

4. **Splitting Data into Train, Test, and Validation Sets**
- **The dataset is divided into three parts:**  
    - `Train (67.5%)` → Used for training the model.  
    - `Validation (22.5%)` → Used for hyperparameter tuning.  
    - `Test (10%)` → Used to evaluate model performance.  
    ```python
    # Make a train, validation and test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y,
    stratify=y, test_size=0.1, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full,
    stratify=y_train_full, test_size=0.25, random_state=0)
    ```
5. **Building the Neural Network Model**
    - The model is designed as a **Deep Neural Network (DNN)**.  
    - It consists of **4 hidden layers**, each with 300 neurons.  
    - **BatchNormalization()** is used to stabilize training.  
    - The **ELU (Exponential Linear Unit)** activation function is applied.  
    - **Dropout (rate = 0.2)** is used to reduce overfitting. 
    
    ```python
    # Set model parameters
    n_layers = 4
    n_nodes = 300
    regularized = False
    dropout = True
    epochs = 50

    # Make a Keras DNN model
    model = keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    for n in range(n_layers):
        if regularized:
            model.add(keras.layers.Dense(n_nodes, kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l1(0.01), use_bias=False))
        else:
            model.add(keras.layers.Dense(n_nodes,
            kernel_initializer="he_normal", use_bias=False))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("elu"))
        if dropout:
            model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy']) 
    ```
6. **Training Configuration and Callbacks**
    - **Loss Function:** `binary_crossentropy` (since this is a binary classification task).  
    - **Optimizer:** `Nadam` (an improved version of Adam).  
    - **Performance Metrics:** `accuracy`, `AUC`, `precision`, `recall`.  
    - **ReduceLROnPlateau:** Reduces the learning rate if `val_loss` does not improve.  
    - **EarlyStopping:** Stops training when the model shows no further improvement.  
    
    ``` python
    # Make a callback that reduces LR on plateau
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                    patience=5, min_lr=0.001)
    # Make a callback for early stopping
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5)
    ```
7. **Model Training**
    - Uses `batch_size = 128` for efficient training.  
    - Trained for **50 epochs**, but stops early if needed.  
    - Training data: `X_train` and `y_train`.  
    - Validation data: `X_valid` and `y_valid`.  
    
    ``` python
    # Train DNN.
    history = model.fit(np.array(X_train), np.array(y_train), epochs=epochs,
     validation_data=(np.array(X_valid), np.array(y_valid)),
      callbacks=[reduce_lr_cb, early_stopping_cb], batch_size=128)
    ```
---
### Summary
This model is designed to predict the winning team (`CT` or `T`) based on round data, including remaining time, score, bomb status, health, and armor of each team. Techniques like Batch Normalization, Dropout, and Early Stopping help stabilize the model and prevent overfitting.  

""")

    st.markdown(""" 
                ---
                
### Evaluation

   - To evaluate a DNN, we'll look at the loss and accuracy scores to see how well training's progressed and check if there's any underfit/overfit. To properly evaluate the model, we'll bring in the yet unseen test set. Afterwards we'll make a few round winner predictions based on the test data. The accuracy for the test set is 80%.
   
   ```python
   model.evaluate(X_test, y_test)
   ```
""")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""### DNN Model
   ```python
   # Make a Keras DNN model
    model = keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    for n in range(n_layers):
        if regularized:
            model.add(keras.layers.Dense(n_nodes, kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l1(0.01), use_bias=False))
        else:
            model.add(keras.layers.Dense(n_nodes,
            kernel_initializer="he_normal", use_bias=False))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("elu"))
        if dropout:
            model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    # Train DNN.
    history = model.fit(np.array(X_train), np.array(y_train), epochs=epochs,
        validation_data=(np.array(X_valid), np.array(y_valid)),
        callbacks=[reduce_lr_cb, early_stopping_cb], batch_size=128)
   ```                    
                    
""")

    with col2:
        st.markdown("""### Random Forest Model
   ```python
    # สร้างโมเดล Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,  # จำนวนต้นไม้ในป่า (default = 100)
        max_depth=10,      # จำกัดความลึกของต้นไม้ (ลด overfitting)
        min_samples_split=5,  # จำนวน sample ขั้นต่ำก่อนแตก node
        min_samples_leaf=2,   # จำนวน sample ขั้นต่ำในใบไม้
        random_state=42,  # ทำให้ผลลัพธ์ reproducible
        n_jobs=-1         # ใช้ CPU ทุก core
    )

    # Train โมเดล
rf_model.fit(X_train, y_train)
   ```                    
                    
""")
        


    
    
elif page == "Machine Learning Model":
    st.title("CS:GO Round Winner Predictor")
    st.write("Predict which team will win the round based on match stats")
    # เลือกโมเดล
    model_option = st.selectbox("Choose a model", ["DNN" ,"Random Forest"])
    # แถบเลื่อนรับข้อมูลจากผู้ใช้
    time_left = st.slider("Time Left (seconds)", 0, 175, 60)

    col1, col2 = st.columns(2)
    with col1:
        ct_score = st.slider("CT Score", min_value=0, max_value=15, value=8)
        ct_health = st.slider("CT Health", min_value=0, max_value=500, value=250)
        ct_armor = st.slider("CT Armor", min_value=0, max_value=500, value=50)
    with col2:
        t_score = st.slider("T Score", min_value=0, max_value=15, value=8)
        t_health = st.slider("T Health", min_value=0, max_value=500, value=250)
        t_armor = st.slider("T Armor", min_value=0, max_value=500, value=50)
    
    # แปลง Map เป็นตัวเลข
    map_dict = {"Dust2": 0, "Mirage": 1, "Inferno": 2, "Nuke": 3, "Overpass": 4, "Vertigo": 5, "Ancient": 6, "Anubis": 7}
    map_selected = st.selectbox("Select Map", list(map_dict.keys()))
    map_encoded = map_dict[map_selected]

    # ปุ่มติ๊ก Bomb Planted (True / False)
    bomb_planted = int(st.checkbox("Bomb Planted"))
    if bomb_planted:
        bomb_planted_True = 1
        bomb_planted_False = 0
    else:
        bomb_planted_True = 0
        bomb_planted_False = 1
        
    if "round_index" not in st.session_state:
        st.session_state["round_index"] = 0  # กำหนดค่าเริ่มต้น
    round_index = st.session_state["round_index"]  # ใช้งานค่าจาก session state
    true_winner = df.iloc[round_index]["round_winner"] if 0 <= round_index < len(df) else "Unknown"# ✅ ดึงค่า `round_winner` ตาม Index ที่เลือก

    if st.button("Predict Team Winner"):
        features = [time_left, ct_score, t_score, ct_health, t_health, ct_armor, t_armor , bomb_planted_False, bomb_planted_True]
        features = np.array(features).reshape(1, -1)     
        
        # เลือกโมเดลที่ใช้
        if model_option == "DNN":
            winner = predict_winner(dnn_model, features)
        else:
            winner = predict_winner(rf_model, features)
        

        # แสดงผล
        col1, col2 = st.columns([1, 2])  
        with col2:
            if winner == "T":
                st.image("T.png", caption="Terrorist", width=150)
            else:
                st.image("CT.png", caption="Counter-Terrorist", width=150)
        with col1:
            st.subheader(f"Predicted Winner: 🏆 {winner}")
            st.subheader(f"True Winner: ✅ {true_winner}")
            if winner == true_winner:
                st.success("🎯 Prediction is Correct!")
            else:
                st.error("❌ Prediction is Incorrect!")
        


elif page == "Neural Network Detail":
    col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

    with col2:
        st.image("csgo.png", caption="Counter-Strike: Global Offensive", width=600)  # Adjust width
    with col1:
        st.title("Neural Network Detail")
        st.markdown("""
            ## CS:GO Introduction""")
    
    st.markdown("""
    **Counter-Strike: Global Offensive (CS:GO)** is a tactical shooter where two teams, Counter-Terrorists (CT) and Terrorists (T), compete in 30 rounds. The first team to win 16 rounds wins the match. 

    - Weapons in CS:GO features a variety of weapons, divided into categories based on their type and role in the game. Each weapon has unique attributes such as damage, fire rate, recoil, and cost.  

    ### **Dataset Preparation for Neural Network Training**
    To train a model to classify CS:GO weapons, we use **[CS:GO Weapon Classification](https://huggingface.co/datasets/Kaludi/data-csgo-weapon-classification)**, available on Hugging Face **Which have 1.1k row of data to train**.
    
    ---
    🔹 Dataset Overview  
         The dataset contains labeled images of **11 CS:GO weapons**, including: 
    ```python
        class_id = predict(image)
        class_labels = ["AK-47", "AWP", "Famas", "Galil-AR", "Glock","M4A1","M4A4","P-90","SG-553","UMP","USP"]
    ```
    ---


    ## 1. Data Preprocessing
    Before training, the dataset undergoes preprocessing to improve model performance:
    - **Resizing**: All images are resized to **224x224 pixels** to match ResNet-18's input size.
    - **Tensor Conversion**: Convert images into PyTorch tensors for model compatibility.
    - **Normalization**: Pixel values are normalized to a range of **[-1, 1]** to help with gradient optimization.
    
        ```python      
        import torch
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize ให้เป็น 224x224
            transforms.ToTensor(),  # แปลงเป็น Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ปรับค่าพิกเซล
        ])
        image_sample = dataset["train"][0]["image"]  # ดึงภาพแรกจาก dataset
        image_tensor = transform(image_sample)  # ใช้ transform
        print(image_tensor.shape)  # ควรได้ (3, 224, 224) เพราะเป็น RGB
        ```
    - **Data Augmentation** (for training set only):
    - **Random Horizontal Flip**: Flips images horizontally to increase diversity.
    - **Random Rotation**: Rotates images within a range of ±10°.
    - **Color Jitter**: Slightly alters brightness and contrast to improve robustness.
    
        ```python      
            # Import transform จาก torchvision
            import torchvision.transforms as transforms
            # Data Augmentation สำหรับ Train Set
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),   # พลิกภาพแนวนอนแบบสุ่ม
                transforms.RandomRotation(10),       # หมุนภาพ ±10 องศา
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # ปรับแสงและ contrast แบบสุ่ม
                transforms.Resize((224, 224)),  # Resize รูปให้ขนาด 224x224
                transforms.ToTensor(),  # แปลงเป็น Tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize ค่า Pixel ให้อยู่ช่วง -1 ถึง 1
            ])

            # สำหรับ Validation และ Test (ไม่มี Augmentation)
            transform_val = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize รูปให้ขนาด 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        ```

    ## 2. Model Architecture: ResNet-18
    We use **ResNet-18**, a deep CNN pre-trained on ImageNet, and modify its final layer:
    - **Dropout Layer (0.5 probability)**: Reduces overfitting by randomly disabling neurons.
    - **Fully Connected Layer**: Adjusted to match the number of CS:GO weapon categories.
    - The model runs on a **GPU** (if available) to speed up training.

        ```python      
        # ใช้ ResNet18 (Pretrained)
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),  # เพิ่ม Dropout ตรงนี้
            nn.Linear(model.fc.in_features, len(label_mapping))
        )
        ```
        
    ## 3. Training Configuration
    - **Loss Function**: `CrossEntropyLoss` - Suitable for multi-class classification.
    - **Optimizer**: `Adam` (learning rate = 0.001, weight decay = 1e-4)
    - **Learning Rate Scheduler**: `ReduceLROnPlateau` (Reduces learning rate when validation loss plateaus)
    - **Early Stopping**: Stops training if the model does not improve after 5 consecutive epochs.

    ## 4. Training Process
    The training pipeline follows these steps:
    1. **Forward Pass**: Images pass through ResNet-18 to generate predictions.
    2. **Loss Computation**: Compare predictions with actual labels using CrossEntropyLoss.
    3. **Backward Pass & Optimization**: Adjust weights using Adam optimizer.
    4. **Validation Phase**: Evaluate model on validation data every epoch.
    5. **Performance Monitoring**: Track training loss, validation loss, and accuracy.
    6. **Reduce Learning Rate (if needed)**: Adjusts learning rate based on validation performance.
    7. **Early Stopping**: Ends training if no improvement is seen for 5 epochs.

        ```python      
        # Early Stopping
        best_val_loss = float('inf')
        early_stop_count = 0
        early_stop_patience = 5  # ปรับตามต้องการ
        # จำนวน epoch
        num_epochs = 20
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # คำนวณ Accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            train_acc = 100 * correct / total
            train_loss = running_loss / len(train_loader)
            # ===== Validation Step =====
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            val_acc = 100 * correct / total
            val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            # ปรับ Learning Rate ถ้า val_loss ไม่ลดลง
            scheduler.step(val_loss)
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_count = 0  # รีเซ็ตถ้าดีขึ้น
            else:
                early_stop_count += 1
            if early_stop_count >= early_stop_patience:
                print("Early stopping triggered!")
                break
                )
        ```

    ## 5. Model Performance & Summary
    - The model is evaluated using accuracy metrics on the training and validation sets.
    - The best model (lowest validation loss) is saved for later testing.
    - Final testing is conducted on unseen data to assess real-world performance.
    
        ```python      
        import matplotlib.pyplot as plt
        # Evaluation on the test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = 100 * correct / total
        test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        # Print some example predictions and labels
        num_examples = 5  # Change to show more or fewer examples
        for i in range(num_examples):
            print(f"Example {i+1}: Predicted Label - {all_predictions[i]}, True Label - {all_labels[i]}")

        # Optionally, you can create a confusion matrix to visualize the model's performance
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        conf_matrix = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()
        )
""")
    

    
elif page == "Neural Network Model":
    st.title("CS:GO Weapon Classifier")
    uploaded_file = st.file_uploader("อัปโหลดภาพปืน CS:GO", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # 🔹 ปรับขนาดภาพให้กว้างสุดไม่เกิน 300 px (รักษาอัตราส่วน)
        max_width = 300
        width, height = image.size
        if width > max_width:
            new_height = int((max_width / width) * height)
            image = image.resize((max_width, new_height))

        # 🔹 แสดงภาพที่ปรับขนาดแล้ว
        
        # ทำการพยากรณ์
        col1, col2 = st.columns(2)
        with col1:
            st.write(" ")
            if st.button("Predict Weapon Type"):
                class_id = predict(image)
                class_labels = ["AK-47", "AWP", "Famas", "Galil-AR", "Glock","M4A1","M4A4","P-90","SG-553","UMP","USP"]  # แก้ให้ตรงกับ label_mapping
                st.write(f"### Prediction: {class_labels[class_id]}")
        with col2:
            st.image(image, caption="Uploaded Image", use_container_width=False)
        

