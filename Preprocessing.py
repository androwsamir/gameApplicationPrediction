import bisect
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import numpy as np
import saveLoadData
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler

class Preprocess:

    saveLoad = saveLoadData.SaveLoadData()
    # Constructor
    def __init__(self, x, y=None):
        self.X = x
        self.Y = y

    def inAppPurchases(self):

        # Take copy of original X
        self.X = self.X.copy()

        # Convert string to List
        self.X['In-app Purchases'] = self.X['In-app Purchases'].apply(lambda x: [float(value) for value in x.split(',')] if isinstance(x, str) else x)

        # Replace the list of values with the median
        self.X['In-app Purchases'] = self.X['In-app Purchases'].apply(lambda x: np.median(x) if x else None)

    def handleDate(self):
        # Take copy of original X
        self.X = self.X.copy()

        # Set Current Version Release Date, Original Release Date to the same format
        self.X['Current Version Release Date'] = pd.to_datetime(self.X['Current Version Release Date'], format='%d/%m/%Y')
        self.X['Original Release Date'] = pd.to_datetime(self.X['Original Release Date'], format='%d/%m/%Y')

        # Split year, month, day of Current Version Release Date every one in new column
        self.X['year'] = self.X['Current Version Release Date'].dt.year
        self.X['Month'] = self.X['Current Version Release Date'].dt.month
        self.X['Day'] = self.X['Current Version Release Date'].dt.day

        # Split year, month, day of Original Release Date every one in new column
        self.X['Oyear'] = self.X['Original Release Date'].dt.year
        self.X['OMonth'] = self.X['Original Release Date'].dt.month
        self.X['ODay'] = self.X['Original Release Date'].dt.day

        # Drop Original Release Date, Current Version Release Date
        self.X.drop('Current Version Release Date', axis=1, inplace=True)
        self.X.drop('Original Release Date', axis=1, inplace=True)

    def handleDescription(self, train=True):
        # Convert ndarray to pandas Series
        data = pd.Series(self.X['Description'])

        if train:
            # Tokenize the text data and create TaggedDocuments
            tagged_data = [TaggedDocument(words=word_tokenize(str(_d).lower()), tags=[str(i)]) for i, _d in
                           enumerate(data)]
            # Train a Doc2Vec model on the TaggedDocuments
            model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=1, workers=4)
            # Save the trained model to a file using pickle
            self.saveLoad.saveModel(model, 'doc2vec_model')
        else:
            # Load the trained model from the file using pickle
            model = self.saveLoad.loadModel('doc2vec_model')

        # Convert each text description into a vector using the trained model
        data = pd.Series(data).apply(
            lambda x: model.infer_vector(word_tokenize(str(x).lower())))
        # Convert the list of vectors to a DataFrame
        new_colDescr = pd.DataFrame(data.to_list(), columns=[f'doc2vec_{i}' for i in range(100)])
        # concatenate dense matrix with other columns
        self.X = pd.concat([self.X.reset_index(drop=True), new_colDescr], axis=1)
        # droop Description column
        self.X.drop('Description', axis=1, inplace=True)

    def handleLanguageGenres(self, columns):

        # Declare dictionary
        oneHotEncode_dict = {}
        for columnName in columns:
            # Apply one-hot encoding to columnName column
            df_encoded = self.X[columnName].str.get_dummies(', ')

            # Save the column names of the encoded dataframe as a list in oneHotEncode_dict
            oneHotEncode_dict[columnName] = df_encoded.columns.tolist()
            # Join the one-hot encoded dataframe back to the original dataframe X
            self.X = pd.concat([self.X, df_encoded], axis=1)
            # Drop columnName from X
            self.X.drop(columnName, axis=1, inplace=True)

        self.oneHotEncode_dict = oneHotEncode_dict

    def handle_outliers(self, columns, data):
        # Handle outliers
        Q1 = self.X[columns].quantile(0.25)
        Q3 = self.X[columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = []
        for col in columns:
            outliers = self.X[
                (self.X[col] < lower_bound[col]) | (self.X[col] > upper_bound[col])].index
            outlier_indices.extend(outliers)
        data.drop(outlier_indices, inplace=True)

        return data

    def dropColumns(self, data):
        dropList = [ 'Subtitle', 'Name', 'ID', 'URL', 'Icon URL', 'Games', 'Strategy', 'Price']
        for element in dropList:
            data.drop(element, axis=1, inplace=True)

        return data

    def encode(self, columns):
        # Take copy of original X
        self.X = self.X.copy()
        # Declare dictionary
        encoder_dict = {}
        for x in columns:
            # Create object of LabelEncoder
            le = LabelEncoder()
            # Learn encoder model on ele of X
            le.fit(self.X[x].values)
            # Implement encode on ele of X
            self.X[x] = le.transform(list(self.X[x].values))
            # save encoder model in dictionary
            encoder_dict[x] = le

        self.encoder_dict = encoder_dict

    def scale_data(self, columns):
        # Create object of MinMaxScaler
        scaler = MinMaxScaler()
        # Implement scaling on columns of X
        self.X[columns] = scaler.fit_transform(self.X[columns])
        self.scaler = scaler

    def trainDataCleaning(self):
        # Handle In-app Purchases
        self.inAppPurchases()
        # ===================================================================================================================================
        # Concatenate X & Y in train_data
        train_data = pd.concat([pd.DataFrame(self.X), pd.DataFrame(self.Y)], axis=1)

        listOutlier = ['User Rating Count'
            , 'In-app Purchases', 'Size']
        # Handle outlier
        train_data = self.handle_outliers(listOutlier, train_data)
        # ===================================================================================================================================
        # Drop duplicate values
        train_data.drop_duplicates(inplace=True)
        # ===================================================================================================================================
        # Split the dataframe train_data back into X and Y
        self.X = train_data.iloc[:, :-1]
        self.Y = train_data.iloc[:, -1]
        # ===================================================================================================================================
        # Handle date column
        self.handleDate()
        # ===================================================================================================================================
        # Handle description column
        self.handleDescription(True)
        # ===================================================================================================================================
        oneHotEncodingList = ['Languages', 'Genres']
        # Handle Languages & Genres columns
        self.handleLanguageGenres(oneHotEncodingList)
        # Save oneHotEncode of Languages & Genres
        self.saveLoad.saveModel(self.oneHotEncode_dict, 'languageGenresEncode')
        # ===================================================================================================================================
        # Drop unique & constant columns
        self.X = self.dropColumns(self.X)
        # ===================================================================================================================================
        # Handle Null Values of In-app Purchases column
        # Calculate Mean of In-app Purchases column
        column_means = self.X['In-app Purchases'].mean()
        # Fill null values with mean
        self.X['In-app Purchases'].fillna(column_means, inplace=True)
        # Save mean value
        self.saveLoad.saveModel(column_means, 'inAppPurchasesMean')
        # ===================================================================================================================================
        # Encode Categorical Data
        listEn = ['Developer', 'Age Rating', 'Primary Genre']
        self.encode(listEn)
        # Save encoder Models
        self.saveLoad.saveModel(self.encoder_dict, 'EncodeValues')
        # ===================================================================================================================================
        # Scale Numerical Data
        listScale = ['User Rating Count', 'In-app Purchases', 'Size']
        self.scale_data(listScale)
        # Save scale model
        self.saveLoad.saveModel(self.scaler, 'scalingValues')

        return self.X, self.Y

    def testDataCleaning(self):
        # Handle In-app Purchases
        self.inAppPurchases()
        # ===================================================================================================================================
        # Handle date column
        self.handleDate()
        # ===================================================================================================================================
        # Handle description column
        self.handleDescription(False)
        # ===================================================================================================================================
        # Load oneHotEncode of Languages & Genres
        oneHotEncode_dict = self.saveLoad.loadModel('languageGenresEncode')
        for columnName in oneHotEncode_dict.keys():
            # get the encoded column names for the current column
            encoded_cols = oneHotEncode_dict.get(columnName)
            # 'Apply one-hot encoding to columnName column'
            df_encoded = self.X[columnName].str.get_dummies(', ')
            df_encoded = df_encoded.reindex(columns=encoded_cols, fill_value=0)
            # Concatenate encoded columns to original dataframe X
            self.X = pd.concat([self.X, df_encoded], axis=1)
            # Drop columnName
            self.X.drop(columnName, axis=1, inplace=True)
        # ===================================================================================================================================
        # Drop unique & constant columns
        self.X = self.dropColumns(self.X)
        # ===================================================================================================================================
        # Load Mean of In-app Purchases column
        columnMean = self.saveLoad.loadModel('inAppPurchasesMean')
        # Fill null values with mean
        self.X['In-app Purchases'].fillna(columnMean, inplace=True)
        # ===================================================================================================================================
        # Load labelEncode_dict
        labelEncode_dict = self.saveLoad.loadModel('EncodeValues')
        for feature in labelEncode_dict.keys():
            encode = labelEncode_dict[feature]
            # Handle unseen values with 'other' value
            self.X[feature] = self.X[feature].map(lambda s: 'other' if s not in encode.classes_ else s)
            le_classes = encode.classes_.tolist()
            bisect.insort_left(le_classes, 'other')
            encode.classes_ = le_classes
            # Implement encode on feature
            self.X[feature] = encode.transform(self.X[feature])
        # ===================================================================================================================================
        # Load scaling model
        scalar = self.saveLoad.loadModel('scalingValues')
        listScale = ['User Rating Count', 'In-app Purchases', 'Size']
        # Apply scaling
        self.X[listScale] = scalar.transform(self.X[listScale])

        return self.X
