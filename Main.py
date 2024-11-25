from pandas import read_csv, DataFrame
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore
from imblearn.combine import SMOTETomek
import joblib



# Assigning column names
dt_colnames = ['age', 'year', 'nodes', 'class']

# Read the dataset
dataset = 'BC.csv'
read_dt = read_csv(dataset, header=None, names=dt_colnames)


# Preprocess the data
input_data, output_data = read_dt.values[:, :-1], read_dt.values[:, -1]
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

# Encode the output classes
input_data = input_data.astype('float32')
output_data = LabelEncoder().fit_transform(output_data)

# Handle class imbalance
smote_tomek = SMOTETomek(random_state=42)
input_data_balanced, output_data_balanced = smote_tomek.fit_resample(input_data, output_data)

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

# Train the model
history = model.fit(in_train, out_train, epochs=400, batch_size=64, validation_data=(in_test, out_test), 
                    callbacks=[checkpoint, early_stopping])

# Save the fitted scaler
joblib.dump(scaler, 'scaler.pkl')

# Evaluate the best saved model
model.load_weights('best_model.keras')
    
out_predict = (model.predict(in_test) > 0.5).astype("int32")
score = accuracy_score(out_test, out_predict)

print(f'Test Accuracy: {round(score, 4)}')
