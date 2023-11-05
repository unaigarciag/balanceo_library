import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

class DataPreprocessing:
    def clean_data(self, df):
        """
        Check if the DataFrame is null or empty, 
        if not, check if there are missing values (NAs) in the DataFrame ,
        If there are NAs, remove the rows with missing values.

        Args:
            df (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        try:
            if df is None or df.empty:
                raise ValueError("El DataFrame es nulo o vacío.")

            if df.isnull().any().any():
                df = df.dropna()
                return df
            else:
                print("No se encontraron valores faltantes en el DataFrame.")
                return df
        except Exception as e:
            print(str(e))
            return None

class DataBalancer(DataPreprocessing):

    def __init__(self, balance_type='undersampling'):
        self.balance_type = balance_type

    def check_libraries(self):
            try:
                import imblearn
                import sdv
            except ImportError:
                print("Faltan bibliotecas necesarias. Asegúrate de que imblearn y sdv estén instalados.")

    def balance_data(self, dataframe, target_column):
        if target_column not in dataframe.columns:
            print("La columna objetivo especificada no existe en el DataFrame.")
        self.target_column = target_column
        preprocessed_df = self.clean_data(dataframe)
        self.preprocessed_df = preprocessed_df
        
        if preprocessed_df is None:
            preprocessed_df = dataframe
            self.preprocessed_df = preprocessed_df
        
        X = dataframe.drop(columns=[target_column])
        y = dataframe[target_column].astype(int)

        if self.balance_type == 'undersampling':
            balanced_X, balanced_y = self.undersampling(X, y)
            return pd.concat([balanced_X, balanced_y], axis=1)
        elif self.balance_type == 'oversampling':
            balanced_X, balanced_y = self.oversampling(X, y)
            return pd.concat([balanced_X, balanced_y], axis=1)
        elif self.balance_type == 'oversampling_sdv':
            df = self.oversampling_sdv()
            return df
        elif self.balance_type == 'mix_sampling':
            balanced_X, balanced_y = self.mix_sampling(X, y)
            return pd.concat([balanced_X, balanced_y], axis=1)
        else:
            raise ValueError("Tipo de balanceo invalido. Elige entre 'undersampling', 'oversampling', 'oversampling_sdv' o'mix_sampling'.")


    def undersampling(self,X,y):
        X = X.copy()
        y = y.copy()

        print('Forma dataset original %s' % Counter(y))        
        nearM = NearMiss()
        X_res, y_res = nearM.fit_resample(X, y)
        print('Forma datased remuestreado %s' % Counter(y_res))
        
        X_res_df = pd.DataFrame(X_res, columns=X.columns)
        y_res_df = pd.Series(y_res, name=self.target_column)

        return X_res_df,y_res_df

    def oversampling(self,X,y):
        
        X = X.copy()
        y = y.copy()

        print('Forma dataset original %s' % Counter(y))        
        randomOS = RandomOverSampler(random_state=7627)
        X_res, y_res = randomOS.fit_resample(X, y)
        print('Forma datased remuestreado %s' % Counter(y_res))
        
        X_res_df = pd.DataFrame(X_res, columns=X.columns)
        y_res_df = pd.Series(y_res, name=self.target_column)

        return X_res_df,y_res_df

    def oversampling_sdv(self):
        df = self.preprocessed_df
        clase_min = pd.DataFrame(df[self.target_column].value_counts().sort_values()).index[0]
        minoritaria = df[df[self.target_column] == clase_min]
        
        n_datos_sinteticos = df[self.target_column].value_counts().sort_values().max()-len(minoritaria)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        metadata.visualize()

        model = CTGANSynthesizer(metadata,
                                 enforce_rounding=False)

        model.fit(minoritaria)

        new_data = model.sample(num_rows=n_datos_sinteticos)

        df_final = pd.concat([df,new_data],ignore_index=True)

        return df_final

        
    def mix_sampling(self,X,y):        
        X = X.copy()
        y = y.copy()

        print('Forma dataset original %s' % Counter(y))        
        smoteT = SMOTETomek(random_state=45678,sampling_strategy=0.678)
        X_res, y_res = smoteT.fit_resample(X, y)
        print('Forma datased remuestreado %s' % Counter(y_res))
        
        X_res_df = pd.DataFrame(X_res, columns=X.columns)
        y_res_df = pd.Series(y_res, name=self.target_column)

        return X_res_df,y_res_df