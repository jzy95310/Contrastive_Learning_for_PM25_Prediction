# Satellite Image Data
## Data Format
This directory should contain the satellite image data for different urban areas along with the corresponding PM2.5 values. The size of the data is too large to be uploaded here. If you want to obtain the image data in order to replicate the experiment in the paper, please contact ziyang.jiang@duke.edu. Or if you want to conduct your own experiment, you can use your own satellite image data as long as it contains the images and the PM2.5 labels and you know the corresponding geographical location and timestamps (used for contrastive learning as described in the next section) for each image as shown below.
```
>> satellite_image_data[0]
{'Image': array([[[251, 239, 221],
         ...,
         [251, 238, 219]],
         
        ...,
        
        [[250, 237, 218],
         ...,
         [249, 236, 216]]], dtype=uint8),
 'PM25': 356.35,
 'Station_index': 'Alipur',
 'Meteo': AT     25.593947
 RH     62.468718
 BP    988.309526
 Name: 2019-11-01 00:00:00, dtype: float64}
```

## Preprocess Data for Contrastive Learning
In addition, to use the satellite image data for contrastive learning, each image should be marked with their corresponding geological location (i.e. latitude and longitude). Also, the images should be grouped by their timestamps. In the experiment described in the paper, the data is preprocessed using the following code:
```
import pickle as pkl

with open('./satellite_image_data.pkl', 'rb') as fp:
    data = pkl.load(fp)
    
stations_latlon_mapping = {
    'Murthal': {'lat': 29.02721, 'lon': 77.06208}, 
    'Arya_Nagar': {'lat': 28.67008, 'lon': 76.92541}, 
    'Pusa_IMD': {'lat': 28.63965, 'lon': 77.14626}, 
    'Shooting_Range': {'lat': 28.49857, 'lon': 77.26484}, 
    'Lodhi_Rd': {'lat': 28.59182, 'lon': 77.22731}, 
    'Aya_Nagar': {'lat': 28.47062, 'lon': 77.10993}, 
    'Sri_Aurobindo_Marg': {'lat': 28.53132, 'lon': 77.19015}, 
    'IGI_Airport_T3': {'lat': 28.56278, 'lon': 77.11801}, 
    'Indirapuram': {'lat': 28.64615, 'lon': 77.3581}, 
    'Najafgarh': {'lat': 28.57012, 'lon': 76.93374}, 
    'Knowledge_ParkV': {'lat': 28.55703, 'lon': 77.45365}, 
    'Patparganj': {'lat': 28.62364, 'lon': 77.28717}, 
    'Sector116': {'lat': 28.56921, 'lon': 77.39384}, 
    'Sector1': {'lat': 28.5898, 'lon': 77.3101}, 
    'Major_Dhyan_Chand_National_Stadium': {'lat': 28.61128, 'lon': 77.23773}, 
    'Sector62': {'lat': 28.62455, 'lon': 77.35771}, 
    'Vikas_Sadan': {'lat': 28.45004, 'lon': 77.02634}, 
    'IHBAS': {'lat': 28.68117, 'lon': 77.30252}, 
    'Mandir_Marg': {'lat': 28.63643, 'lon': 77.20107}, 
    'Knowledge_ParkIII': {'lat': 28.47273, 'lon': 77.48199}, 
    'Sanjay_Nagar': {'lat': 28.68539, 'lon': 77.45383}, 
    'NISE_Gwal_Pahari': {'lat': 28.42267, 'lon': 77.14893}, 
    'New_Collectorate': {'lat': 28.97479, 'lon': 77.21335}, 
    'Sirifort': {'lat': 28.55042, 'lon': 77.21594}, 
    'Okhla_Phase2': {'lat': 28.53072, 'lon': 77.27121}, 
    'North_Campus': {'lat': 28.65738, 'lon': 77.15854}, 
    'R_K_Puram': {'lat': 28.56326, 'lon': 77.18694}, 
    'Sonia_Vihar': {'lat': 28.71032, 'lon': 77.24945}, 
    'Loni': {'lat': 28.75728, 'lon': 77.27879}, 
    'Vivek_Vihar': {'lat': 28.67229, 'lon': 77.31532}, 
    'Dwarka_Sector_8': {'lat': 28.57099, 'lon': 77.07193},
    'Shadipur': {'lat': 28.65148, 'lon': 77.14731}, 
    'CRRI_MTR_Rd': {'lat': 28.5512, 'lon': 77.27357},
    'ITO': {'lat': 28.62855, 'lon': 77.24102}, 
    'Alipur': {'lat': 28.81606, 'lon': 77.15266}, 
    'Narela': {'lat': 28.8227, 'lon': 77.10191}, 
    'Sector16A': {'lat': 28.40884, 'lon': 77.30988}, 
    'NSIT_Dwarka': {'lat': 28.60902, 'lon': 77.03251}, 
    'Ashok_Vihar': {'lat': 28.69538, 'lon': 77.18163}, 
    'Punjabi_Bagh': {'lat': 28.67405, 'lon': 77.13102}, 
    'Sector125': {'lat': 28.54476, 'lon': 77.32313}, 
    'Nehru_Nagar': {'lat': 28.56786, 'lon': 77.25046}, 
    'Burari_Crossing': {'lat': 28.72556, 'lon': 77.20111}, 
    'DTU': {'lat': 28.75005, 'lon': 77.11126}, 
    'Bawana': {'lat': 28.77618, 'lon': 77.0511}, 
    'Rohini': {'lat': 28.73251, 'lon': 77.11993}, 
    'Vasundhara': {'lat': 28.66033, 'lon': 77.35726}, 
    'Jahangirpuri': {'lat': 28.73278, 'lon': 77.17064}, 
    'Mundka': {'lat': 28.68449, 'lon': 77.07668}, 
    'Wazirpur': {'lat': 28.69972, 'lon': 77.1654}, 
    'Anand_Vihar': {'lat': 28.6469, 'lon': 77.31592}, 
}

for data_point in data:
    data_point['lat'] = stations_latlon_mapping[data_point['Station_index']]['lat']
    data_point['lon'] = stations_latlon_mapping[data_point['Station_index']]['lon']

timestamps = set(map(lambda x: x['Meteo'].name, data))
data = [[data_point for data_point in data if data_point['Meteo'].name == time] for time in timestamps]

with open('./satellite_image_data_preprocessed.pkl', 'wb') as fp:
    pkl.dump(data, fp, protocol=pkl.HIGHEST_PROTOCOL)
```
