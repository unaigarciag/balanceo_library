import pandas as pd
import numpy as np

class DataPreprocessing:
    def clean_data(self, df):
        try:
            # Verificar si el DataFrame es nulo o vacío
            if df is None or df.empty:
                raise ValueError("El DataFrame es nulo o vacío.")

            # Verificar si hay valores faltantes (NAs) en el DataFrame
            if df.isnull().any().any():
                # Si hay NAs, eliminar las filas con valores faltantes
                df = df.dropna()
                return df
            else:
                raise ValueError("No se encontraron valores faltantes en el DataFrame.")

        except Exception as e:
            print(f"Error al limpiar datos: {str(e)}")
            return None

class DataBalancer(DataPreprocessing):

    def __init__(self, balance_type='undersampling'):
        self.balance_type = balance_type

    def balance_data(self, dataframe, target_column):
        self.target_column = target_column
        preprocessed_df = self.clean_data(dataframe)
        self.preprocessed_df = preprocessed_df
        # df[self.target_column] = df[self.target_column].astype(int)
        # Si preprocessed_df es None, utilizar el DataFrame original
        if preprocessed_df is None:
            preprocessed_df = dataframe
            self.preprocessed_df = preprocessed_df
        X = preprocessed_df.drop(columns=[target_column])
        y = preprocessed_df[target_column].astype(int)

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
            raise ValueError("Invalid balance_type. Choose from 'undersampling', 'oversampling', or 'mix_sampling'.")

        


    def undersampling(self,X,y):

        from collections import Counter
        from imblearn.under_sampling import NearMiss

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

        from imblearn.over_sampling import RandomOverSampler
        from collections import Counter
        
        X = X.copy()
        y = y.copy()

        print('Forma dataset original %s' % Counter(y))        
        nearM = RandomOverSampler(random_state=7627)
        X_res, y_res = nearM.fit_resample(X, y)
        print('Forma datased remuestreado %s' % Counter(y_res))
        
        X_res_df = pd.DataFrame(X_res, columns=X.columns)
        y_res_df = pd.Series(y_res, name=self.target_column)

        return X_res_df,y_res_df

    def oversampling_sdv(self):
        import os
        import time
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer,CTGANSynthesizer

        df = self.preprocessed_df
        clase_min = pd.DataFrame(df[self.target_column].value_counts().sort_values()).index[0]
        minoritaria = df[df[self.target_column] == clase_min]
        
        n_datos_sinteticos = df[self.target_column].value_counts().sort_values().max()-len(minoritaria)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        metadata.visualize()

        model = CTGANSynthesizer(metadata,
                                 enforce_rounding=False)
                                        #default_distribution='norm')

        model.fit(minoritaria)

        new_data = model.sample(num_rows=n_datos_sinteticos)

        df_final = pd.concat([df,new_data],ignore_index=True)
        # print(df_final.shape)
        # print(n_datos_sinteticos)
        # print(df[self.target_column].value_counts().sort_values().max())

        return df_final

        
    def mix_sampling(self,X,y):
        from imblearn.combine import SMOTETomek
        from collections import Counter
        
        X = X.copy()
        y = y.copy()

        print('Forma dataset original %s' % Counter(y))        
        nearM = SMOTETomek(random_state=45678,sampling_strategy=0.678)
        X_res, y_res = nearM.fit_resample(X, y)
        print('Forma datased remuestreado %s' % Counter(y_res))
        
        X_res_df = pd.DataFrame(X_res, columns=X.columns)
        y_res_df = pd.Series(y_res, name=self.target_column)

        return X_res_df,y_res_df

if __name__ == '__main__':
    df = pd.read_csv('C:/Users/Asus/Documents/Mondragon/bdata4año/Programacion/Trabajo_iker_unai_bueno/Thyroids.csv')
    datos = DataBalancer('oversampling_sdv')
    df_balanceado = datos.balance_data(df,'clase')
    df_balanceado