from pandas import read_csv
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score , classification_report
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
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
    print(f'Class={key}, Count={value}, Percent={round(percent,4)}%')

# Preprocess the data
input_data, output_data = read_dt.values[:, :-1], read_dt.values[:, -1]
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

# Encode the output classes
input_data = input_data.astype('float32')
output_data = LabelEncoder().fit_transform(output_data)

# Train Test Split
#in_train, in_test, out_train, out_test = train_test_split(input_data, output_data, test_size=0.20, random_state=1, stratify=output_data)

# Number of features
n_features = input_data.shape[1]
smote = SMOTE(random_state=42)
input_data_balanced, output_data_balanced = smote.fit_resample(input_data, output_data)

# Train-Test Split
in_train, in_test, out_train, out_test = train_test_split(
    input_data_balanced, output_data_balanced, test_size=0.20, random_state=1
)

# Build a deeper model
model = Sequential()
model.add(Dense(512, activation='leaky_relu', input_shape=(n_features,), kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='leaky_relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(128, activation='leaky_relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='leaky_relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.0009)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(in_train, out_train, epochs=260, batch_size=64, validation_data=(in_test, out_test), callbacks=[checkpoint, early_stopping])

# Evaluate the best saved model
model.load_weights('best_model.keras')
out_predict = (model.predict(in_test) > 0.5).astype("int32")
score = accuracy_score(out_test, out_predict)

print(f'Improved Test Accuracy: {round(score, 4)}')
print(classification_report(out_test, out_predict))