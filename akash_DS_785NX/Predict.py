"""
This Python code snippet serves two purposes:
1. illustrates how to use relative path
2. provides the template for code submission
ASSUMPTION: 
1. This Python code is present in the folder 'srika_DS_456AB'.
2. BMTC.parquet.gzip, Input.csv, and GroundTruth.csv are present in the folder 'data'
"""
import pandas as pd
# import other packages here
from math import radians, cos, sin, asin, sqrt, atan2, degrees
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

"""
ILLUSTRATION: HOW TO USE RELATIVE PATH
Given the above mentioned assumptions, when you run the code, the following three commands will read the files 
containing data, input and, the ground truth.
"""
df = pd.read_parquet('./../data/BMTC.parquet.gzip', engine='pyarrow') # This command loads BMTC data into a dataframe. 
                                                                      # In case of error, install pyarrow using: 
                                                                      # pip install pyarrow
dfInput = pd.read_csv('./../data/Input.csv')
dfGroundTruth = pd.read_csv('./../data/GroundTruth.csv') 
# NOTE: The file GroundTruth.csv is for participants to assess the performance their own codes

"""
CODE SUBMISSION TEMPLATE
1. The submissions should have the function EstimatedTravelTime().
2. Function arguments:
    a. df: It is a pandas dataframe that contains the data from BMTC.parquet.gzip
    b. dfInput: It is a pandas dataframe that contains the input from Input.csv
3. Returns:
    a. dfOutput: It is a pandas dataframe that contains the output
"""
def EstimatedTravelTime(df, dfInput): # The output of this function will be evaluated
    # Function body - Begins
    # Make changes here.
    dfOutput = pd.DataFrame()
    model=training_model(df)

    sc=StandardScaler()
    X_test=sc.fit_transform(dfInput)
    estimated_time=model.predict(X_test)
    output=dfInput.copy()
    output["ETT"]=round(estimated_time,2)

    dfOutput=dfOutput.append(output)
    # Function body - Ends
    return dfOutput 
  
"""
Other function definitions here: BEGINS
"""
def training_model(data):
  trained_dataset=prepare_dataset(data.copy())
  X = df_train.drop('Time', axis=1)
  y=df_train['Time']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  sc=StandardScaler()
  X_train=sc.fit_transform(X_train)
  X_test=sc.transform(X_test)

  model = DecisionTreeRegressor(random_state=100)
  model.fit(X_train, y_train)

  predictions = model.predict(X_test)

  return model

def prepare_dataset(data):
  df = data.drop('Timestamp',axis=1)
  df.rename(columns = {'Latitude':'Source_Lat', 'Longitude':'Source_Long','Speed':'Initial_Speed'}, inplace = True)
  df.insert(loc=3,column='Dest_Long',value='')
  df.insert(loc=3,column='Dest_Lat',value='')
  df.insert(loc=6,column='Final_Speed',value='')
  df.insert(loc=7,column='Distance',value='')
  df.insert(loc=8,column='Calculated Time',value='')

  df["Dest_Lat"]=df["Source_Lat"].shift(-1)
  df["Dest_Long"]=df["Source_Long"].shift(-1)
  df["Final_Speed"]=df["Initial_Speed"].shift(-1)

  df=df.dropna()

  df=df.apply(calculate_data,axis=1)
  df = df.drop(['BusID','Initial_Speed','Final_Speed','Distance'],axis=1)
  return df


def calculate_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth.
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def calculate_data(row):
  Dest_Lat=row["Dest_Lat"]
  Dest_Long=row["Dest_Long"]
  Source_Lat=row["Source_Lat"]
  Source_Long=row["Source_Long"]
  distance = calculate_distance(Dest_Long, Dest_Lat, Source_Long, Source_Lat)
  u=row["Initial_Speed"]
  v=row["Final_Speed"]
  row["Distance"]=distance
  if distance!=0:
    a=((v*v)-(u*u))/(2*distance)
  else:
    a=0
  if a!=0:
    t=(v-u)/a
    row["Calculated Time"]=t*60
  else:
    row["Calculated Time"]=0
  return row

"""
Other function definitions here: ENDS
"""

dfOutput = EstimatedTravelTime(df, dfInput)