import pandas as pd
import numpy as np

class DataBalancer():

    def __init__(self, balance_type='undersampling'):
        self.balance_type = balance_type

    def data_preprocessing(self,df):
        if df.isna().any().any():
            df = df.dropna()
        return df

    def balance_data(self,dataframe,target_column):

        df_preprocessed = self.data_preprocessing(dataframe)

        X = df_preprocessed.drop(columns=[target_column])
        y = df_preprocessed[target_column]

        #We have to check what kind of balance the person wants to do.
        if self.balance_type == 'undersampling':
            balanced_X, balanced_y = self.undersampling(X, y)
        elif self.balance_type == 'oversampling':
            balanced_X, balanced_y = self.oversampling(X, y)
        elif self.balance_type == 'oversampling_sdv':
            balanced_X, balanced_y = self.oversampling_sdv(X, y)
        elif self.balance_type == 'mix_sampling':
            balanced_X, balanced_y = self.mix_sampling(X, y)
        else:
            raise ValueError("Invalid balance_type. Choose from 'undersampling', 'oversampling', or 'mix_sampling'.")
        
        return pd.concat([balanced_X,balanced_y],axis=1)


    def undersampling(self,X,y):

        from sklearn.svm import LinearSVC
        from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay,confusion_matrix

        from imblearn.under_sampling import NearMiss
        from imblearn.pipeline import make_pipeline
        from imblearn.metrics import classification_report_imbalanced

        X = X.copy()
        y = y.copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        undersampler = NearMiss(sampling_strategy='auto',version=2)
        classifier = LinearSVC()

        model = make_pipeline(undersampler, classifier)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        # cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        print(classification_report_imbalanced(y_test, y_pred))
        print('AUC = ' + str(roc_auc_score(y_test, y_pred)))
    
        return X,y
    def oversampling(self,X,y):

        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import RandomOverSampler
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        os =  RandomOverSampler(ratio=0.5)
        X_train_res, y_train_res = os.fit_sample(X_train, y_train)
        return X_train_res, y_train_res

    def oversampling_sdv(self,X,y):
        pass

    def mix_sampling(self,X,y):
        pass

if __name__ == '__main__':
    df = pd.read_csv('C:/Users/Asus/Documents/Mondragon/bdata4año/Programacion/Trabajo_grupal/Thyroids.csv')
    datos = DataBalancer('undersampling')
    datos.balance_data(df,'clase')
            