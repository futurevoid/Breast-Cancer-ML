from pandas import read_csv, DataFrame
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore
from imblearn.over_sampling import SMOTE

# Assigning column names
dt_colnames = ['age', 'year', 'nodes', 'class']

# Read the dataset
dataset = 'BC.csv'
read_dt = read_csv(dataset, header=None, names=dt_colnames)
print(read_dt)

# Calculate target distribution
targets = read_dt['class'].values
target_counter = Counter(targets)
for key, value in target_counter.items():
    percent = value / len(targets) * 100
    print(f'Class={key}, Count={value}, Percent={round(percent, 4)}%')

# Preprocess the data
input_data, output_data = read_dt.values[:, :-1], read_dt.values[:, -1]
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

# Encode the output classes
input_data = input_data.astype('float32')
output_data = LabelEncoder().fit_transform(output_data)

# Handle class imbalance
smote = SMOTE(random_state=42)
input_data_balanced, output_data_balanced = smote.fit_resample(input_data, output_data)

# Train-Test Split
in_train, in_test, out_train, out_test = train_test_split(
    input_data_balanced, output_data_balanced, test_size=0.20, random_state=1
)

# Build a deeper model
model = Sequential()
model.add(Dense(256, activation='leaky_relu', input_shape=(input_data.shape[1],), kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='leaky_relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='leaky_relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='leaky_relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.0009)  # Lower learning rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

mode = input('train or predict?')
if mode == 'train':
    # Train the model
    history = model.fit(in_train, out_train, epochs=400, batch_size=64, validation_data=(in_test, out_test), 
                        callbacks=[checkpoint, early_stopping])

    # Evaluate the best saved model
    model.load_weights('best_model.keras')
    out_predict = (model.predict(in_test) > 0.5).astype("int32")
    score = accuracy_score(out_test, out_predict)

    print(f'Improved Test Accuracy: {round(score, 4)}')
    print(classification_report(out_test, out_predict))
elif mode == 'predict':
    # Patient input
    age = float(input("Enter patient's age: "))
    year = float(input("Enter patient's year: "))
    nodes = float(input("Enter patient's nodes: "))

    # Example single input data
    single_input = {'age': age, 'year': year, 'nodes': nodes}

    # Convert the single input into a NumPy array
    single_input_array = np.array([list(single_input.values())], dtype=np.float32)

    # Apply scaling to match the training process
    single_input_scaled = scaler.transform(single_input_array)

    # Debugging: Output scaled input
    print(f"Scaled Input: {single_input_scaled}")

    # Make the prediction
    prediction_prob = model.predict(single_input_scaled)[0, 0]  # Probability output
    prediction = (prediction_prob > 0.5).astype("int32")  # Threshold

    # Debugging: Output prediction probability
    print(f"Prediction Probability: {prediction_prob}")

    # Interpret the prediction result
    survival_status = "Survived 5 years or more" if prediction == 1 else "Survived less than 5 years"
    print(f"Predicted Survival Status: {survival_status}")
else:
    print("Invalid mode. Please enter 'train' or 'predict'.")


