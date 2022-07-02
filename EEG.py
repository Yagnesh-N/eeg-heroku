import numpy as np
import pandas as pd
import pickle

from flask import Flask, render_template, request

app = Flask(__name__)

#########################################################################################################################################################

def padding(grp):
        
    # Drop unnecessary columns : SubjectID', 'VideoID','predefinedlabel' and 'userdefinedlabel'
    grp = grp.drop(['SubjectID', 'VideoID', 'predefinedlabel', 'userdefinedlabel'], axis = 1)

    # maximum video duration
    max_length = 144

    grp_length = grp.shape[0]

    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    padding_list = [0]* grp.shape[1]

    # number of rows to be padded for each student-video data
    num_padding_rows = max_length - grp_length

    # Add padding rows to those grp whose max_length < 144.
    padding_array = pd.DataFrame([padding_list] * num_padding_rows, columns = grp.columns)

    # concatinating
    grp = pd.concat([grp, padding_array], axis = 0, ignore_index = True)

    # numpy array 
    grp = grp.values

    return grp

#########################################################################################################################################################

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling1D

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam


def model():
    
    Model_1 = Sequential()
    
    Model_1.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (144, 11)))
    
    Model_1.add(MaxPooling1D(pool_size = 2))
    
    Model_1.add(LSTM(units = 100, return_sequences = False))
    
    Model_1.add(Dropout(0.04))
    
    Model_1.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Training the LSTM
    Model_1.compile(optimizer = Adam(),
                    loss      = 'binary_crossentropy',
                    metrics   = ['accuracy'])

    # weights loading
    Model_1.load_weights('model_weights/Model_weights.hdf5')

    return Model_1

#########################################################################################################################################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/test" , methods = ['GET', 'POST'])

def test():
    
    #
    data_file = request.files['file']

    df = pd.read_csv(data_file, index_col = 0)
     
    # This loads your dict
    min_max_dict = pickle.load(open('pickle_files/min_max_Range.pkl','rb'))

    if df.loc[:, ['Attention']].eq(0).any().any() + df.loc[:, ['Mediation']].eq(0).any().any():
        return 'Attention and Mediation Outlier !! Enter valid values...'

    elif df.loc[:, ['Raw']].lt(min_max_dict['Raw_min']).any().any() + df.loc[:, ['Raw']].gt(min_max_dict['Raw_max']).any().any():
        return 'Raw Outlier !! Enter valid values...'

    elif df.loc[:, ['Delta']].lt(min_max_dict['Delta_min']).any().any() + df.loc[:, ['Delta']].gt(min_max_dict['Delta_max']).any().any():
        return 'Delta Outlier !! Enter valid values...'

    elif df.loc[:, ['Theta']].lt(min_max_dict['Theta_min']).any().any() + df.loc[:, ['Theta']].gt(min_max_dict['Theta_max']).any().any():
        return 'Theta Outlier !! Enter valid values...'

    elif df.loc[:, ['Alpha1']].lt(min_max_dict['Alpha1_min']).any().any() + df.loc[:, ['Alpha1']].gt(min_max_dict['Alpha1_max']).any().any():
        return 'Alpha1 Outlier !! Enter valid values...'

    elif df.loc[:, ['Alpha2']].lt(min_max_dict['Alpha2_min']).any().any() + df.loc[:, ['Alpha2']].gt(min_max_dict['Alpha2_max']).any().any():
        return 'Alpha2 Outlier !! Enter valid values...'

    elif df.loc[:, ['Beta1']].lt(min_max_dict['Beta1_min']).any().any() + df.loc[:, ['Beta1']].gt(min_max_dict['Beta1_max']).any().any():
        return 'Beta1 Outlier !! Enter valid values...'

    elif df.loc[:, ['Beta2']].lt(min_max_dict['Beta2_min']).any().any() + df.loc[:, ['Beta2']].gt(min_max_dict['Beta2_max']).any().any():
        return 'Beta2 Outlier !! Enter valid values...'

    elif df.loc[:, ['Gamma1']].lt(min_max_dict['Gamma1_min']).any().any() + df.loc[:, ['Gamma1']].gt(min_max_dict['Gamma1_max']).any().any():
        return 'Gamma1 Outlier !! Enter valid values...'

    elif df.loc[:, ['Gamma2']].lt(min_max_dict['Gamma2_min']).any().any() + df.loc[:, ['Gamma2']].gt(min_max_dict['Gamma2_max']).any().any():
        return 'Gamma2 Outlier !! Enter valid values...'

    else:
        
        # padding
        X = np.array(df.groupby(['SubjectID','VideoID']).apply(padding).values.tolist()) # (1, 144, 11)

        X = np.reshape(X, (1, 144 * 11))
        
        X = np.array(X)
        
        # StandardScaler model
        Standard_Scaler = pickle.load(open('pickle_files/Standard_Scaler.pkl','rb'))
    
        # StandardScaler model
        X = Standard_Scaler.transform(X)

        X = np.reshape(X, (1, 144, 11))

        X = np.asarray(X).astype('float32')

        # model Bi-LSTM
        model_df = model()

        # predict
        y_pred = model_df.predict(X)


        # Conclusion
        if y_pred > 0.5:
            return 'The student seems confused while watching the video'

        else:
            return 'The student understood everything... So no confusion while watching the video'
  
#########################################################################################################################################################

if __name__=='__main__':

    app.run(debug = True)
