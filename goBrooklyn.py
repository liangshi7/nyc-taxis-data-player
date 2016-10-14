import matplotlib
matplotlib.use('Agg')
import seaborn as sns
matplotlib.rcParams['savefig.dpi'] = 2 * matplotlib.rcParams['savefig.dpi']


import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas
import os


class nycTaxiDataPlayer(object):
    '''This class handles both raw csv files and the csv files pre-cleaned by nycTaxiDataCleaner()
    '''

    def __init__(self):
        return
        
    def read_tidy_csv(self, filename, MAX_ROWS=3000000):
        ''' read csv file and pre-process the data
        '''
        self.filename = os.path.realpath(filename)

        # check raw data or pre-cleaned data
        columns = os.popen('head -1 %s'%self.filename).read()
        self.isRaw = ('borough' not in columns)

        # read the data and remove invalid position entries
        raw = pd.read_csv(filename,parse_dates=[1,2],infer_datetime_format=True,nrows=MAX_ROWS)
        if self.isRaw:
            raw = raw.rename(columns=lambda col: col.lower().replace('lpep_','').replace('tpep_',''))
            goodLat = (raw.pickup_latitude > 40.5) & (raw.pickup_latitude < 41
                                                     ) & (raw.dropoff_latitude > 40.5) & (raw.dropoff_latitude < 41)
            goodLong = (raw.pickup_longitude > -74.3) & (raw.pickup_longitude < -73.7
                                                     ) & (raw.dropoff_longitude > -74.3) & (raw.dropoff_longitude < 73.7)
            raw = raw[goodLat].loc[goodLong]
        
        # calculate the orientation of each trip
        raw = self.get_angle(raw)
        
        return raw
    
    def get_angle(self,df):
        '''A funtion to calculate the point-to-point orientation of the trips
        Angle (with respect to the East) range: [-90, 270]
        '''
        if 'angle' in df.columns:
            return df
        
        angleExpr = 'angle = 180*(dropoff_longitude < pickup_longitude)+'
        angleExpr += '180/3.14159*arctan((dropoff_latitude - pickup_latitude)/(dropoff_longitude - pickup_longitude)/0.77)'
        df.eval(angleExpr, inplace=True)
        
        return df    
        
    def to_geo(self, data, ifMerge=False, pickup = True):
        '''A function to transform the data frame to GeoDataFrame. 
        If ifMerge == True, it will add the borough and neighbor using the lat and long
        '''                
        df = data.copy()  
        
        # creat geography point
        word = 'pickup' if pickup else 'dropoff'
        df['geometry'] = map(Point, df[['%s_longitude'%word,'%s_latitude'%word]].values) # 4x faster than .apply
        df_geo = geopandas.GeoDataFrame(df, geometry='geometry')
        df_geo.crs = {'init': 'epsg:4326'}

        # get borough and neighborhood columns by sjoin-ing the shape file
        if ifMerge:
            nycZones = geopandas.read_file('data/nycneighborhoods.geojson')
            df_geo = geopandas.sjoin(df_geo, nycZones, how="left", op='within') # using'within' speedup O(10^2)
            df_geo = df_geo.rename(columns={'borough':'%s_borough'%word, 'neighborhood':'%s_neighborhood'%word})
            df_geo.drop(['@id','index_right','geometry','boroughCode'],axis=1,inplace=True)

        return df_geo
    
    def round_spacetime(self, data, sigDigits=2, freq='10min', ifAngle=False):
        '''A function to round/bin the space and time of pickups and dropoffs, 
        e.g., to study the ride aggregations. 
        
        sigDigits and freq are the pickups' space and time resolution
        '''
        df = data.copy()

        # round pickup time to the nearest 5min, space to the thousandth digit (~0.25 nyc block length)
        # precisions are fixed for pickup since customers are less flexible for the pickup
        colname = 'pickup'
        df['%s_datetime'%colname] = df['%s_datetime'%colname].dt.round(freq) # 10^n times faster than .apply method
        df['%s_latitude'%colname] = df['%s_latitude'%colname].round(sigDigits)
        df['%s_longitude'%colname] = df['%s_longitude'%colname].round(sigDigits)
        
        # round dropoff 
        colname = 'dropoff'        
        if ifAngle:
            df['angle'] = df['angle'].round(-1)
        else:
            df['%s_latitude'%colname] = df['%s_latitude'%colname].round(2)
            df['%s_longitude'%colname] = df['%s_longitude'%colname].round(2)

        return df
    
    def bin_series(self, series, nbins=700, resol=0.001):
        '''A function to bin the numeric series by resolution
        '''
        smin = series.min()
        bins = [smin + resol*i for i in xrange(nbins)]
        return pd.cut(series,bins=bins)
    
    def get_reduced_rides(self, data, resol=0.005):
        '''A function to calculate the percentage of reduced rides 
        '''
        df = data.copy()
                
        # binning 
        df['pickup_long_bin'] = self.bin_series(df.pickup_longitude,resol=resol)
        df['pickup_lat_bin'] = self.bin_series(df.pickup_latitude,resol=resol)
        df['pickup_dt_bin'] = df.pickup_datetime.dt.round('5min')
        df['angle_bin'] = df['angle'].round(-1)
        df['dropoff_long_bin'] = self.bin_series(df.dropoff_longitude,resol=0.01)
        df['dropoff_lat_bin'] = self.bin_series(df.dropoff_latitude,resol=0.01)

        # calculate stats  
        n_tot = float(len(df)) 
        passenger_count_median = df.passenger_count.median()
        total_amount_median = df.total_amount.median()
        trip_distance_median = df.trip_distance.median()
        cols = ['pickup_long_bin','pickup_lat_bin','pickup_dt_bin']
        
        # economic savings for point-point aggregation
        point_cols = cols + ['dropoff_long_bin','dropoff_lat_bin']
        n_point = df.groupby(point_cols).size().size
        trip_saved_point = n_tot-n_point
        savings = [trip_saved_point]
        savings.append(trip_saved_point/n_tot)
        savings.append(trip_saved_point*passenger_count_median)
        savings.append(trip_saved_point*total_amount_median)
        savings.append(trip_saved_point*trip_distance_median)
        
        # economic savings for point-angle aggregation
        angle_cols = cols + ['angle_bin']
        n_angle = df.groupby(angle_cols).size().size
        trip_saved_angle = n_tot-n_angle
        savings.append(trip_saved_angle)
        savings.append(trip_saved_angle/n_tot)
        savings.append(trip_saved_angle*passenger_count_median)
        savings.append(trip_saved_angle*total_amount_median)
        savings.append(trip_saved_angle*trip_distance_median)

        return tuple(savings)
               
    def plot_week_hour(self, df):
        '''Plot the saved number and percentage of trips by weekdays and hours
        '''
        week_hour = df.groupby([df.pickup_datetime.dt.weekday_name,df.pickup_datetime.dt.hour])
        week_hour_savings = week_hour.apply(self.get_reduced_rides)

        cols = ['ntrips_pp','percent_pp','passengers_pp','dollars_pp','mileage_pp']
        cols.extend(['ntrips_pa','percent_pa','passengers_pa','dollars_pa','mileage_pa'])
        week_hour_savings=pd.DataFrame.from_records(list(week_hour_savings.values),index=week_hour_savings.index,columns=cols)
        week_hour_savings.index.names = ['Weekday','Hour']

        # plot the figures
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
        week_hour_savings.unstack(level=0)['ntrips_pp'].plot(marker='o',ax=ax1)
        week_hour_savings.unstack(level=0)['ntrips_pa'].plot(marker='o',ax=ax2,legend=False)
        week_hour_savings.unstack(level=0)['percent_pp'].plot(marker='o',ax=ax3,legend=False)
        week_hour_savings.unstack(level=0)['percent_pa'].plot(marker='o',ax=ax4,legend=False)
        ax1.set_title('point-point aggregation')
        ax2.set_title('point-angle aggregation')
        ax3.set_xlabel('Time in a day [hour]')
        ax1.set_ylabel('Number of saved rides')
        ax4.set_xlabel('Time in a day [hour]')
        ax3.set_ylabel('Percent of saved rides')

        return week_hour_savings, fig
    
    def plot_top_pickups_dropoffs(self,df,n):
        '''A function to plot the top n locations for pickup and dropoff in Brooklyn
        '''
        df_round = self.round_spacetime(df) # round to the nearest 0.2 miles
        pickup_hot_spots = df_round.query('pickup_borough=="Brooklyn"').groupby(['pickup_longitude','pickup_latitude']
                                                                               ).size().nlargest(n).reset_index()
        pickup_hot_spots.rename(columns={0:'counts'},inplace=True)
        dropoff_hot_spots = df_round.query('dropoff_borough=="Brooklyn"').groupby(['dropoff_longitude','dropoff_latitude']
                                                                                ).size().nlargest(n).reset_index()
        dropoff_hot_spots.rename(columns={0:'counts'},inplace=True)
        
        cmax = max(pickup_hot_spots.counts.max(),dropoff_hot_spots.counts.max())
        nycZones = geopandas.read_file('data/nycneighborhoods.geojson')
        fig,ax=plt.subplots()
        nycZones.plot('borough',ax=ax,edgecolor='w',linewidth=0.2,alpha=0.3)
        pickup_hot_spots.plot(kind='scatter',x='pickup_longitude',y='pickup_latitude',
                              s=pickup_hot_spots.counts*50/cmax,ax=ax,c='b',edgecolors='none')
        dropoff_hot_spots.plot(kind='scatter',x='dropoff_longitude',y='dropoff_latitude',
                               s=dropoff_hot_spots.counts*50/cmax,ax=ax,c='r',edgecolors='none')
        title = 'Top %d pickup (blue) and dropoff (red) locations in Brooklyn\n '%n
        title += 'The size of the circle represents the relative popularity'
        ax.set_title(title)
        ax.axis('off')
        
        return fig,ax
        
                
def combine_yellow_green(yellow,green):
    '''A function to combine the yellow and green taxi data frame 
    '''
    cols = filter(lambda cn: ('pickup' in cn) or ('dropoff' in cn),green.columns.values)
    cols.extend(['passenger_count','angle','total_amount','trip_distance'])
    
    return yellow[cols].append(green[cols])

def transform_nyc_taxi_file(filename):
    '''A function to clean the NYC green or yellow taxi trip data and add borough columns
    '''
    color = 'green' if 'green' in filename else 'yellow'
    nyc = nycTaxiDataPlayer()
    data = nyc.read_tidy_csv(filename,MAX_ROWS=100)
    data_geo = nyc.to_geo(data,ifMerge=True)
    data_geo = nyc.to_geo(data_geo,ifMerge=True,pickup=False)
    data_geo.to_csv('%s.csv'%color,index=False)
    

def brooklyn_expansion_analysis():
    '''brooklyn_expansion_analysis'''
    
    nyc = nycTaxiDataPlayer()
    
    # read the file
    green = nyc.read_tidy_csv('data/green.csv')#, MAX_ROWS=10000)
    yellow = nyc.read_tidy_csv('data/yellow.csv')#, MAX_ROWS=10000)
    
    # limit trips within and between Manhattan and Brooklyn
    queryExpr = '(pickup_borough == "Manhattan" | pickup_borough == "Brooklyn") & '
    queryExpr += '(dropoff_borough == "Manhattan" | dropoff_borough == "Brooklyn")'
    yelgreen = combine_yellow_green(green.query(queryExpr),yellow.query(queryExpr))

    # number of trips 
    boro = yelgreen.groupby(['pickup_borough','dropoff_borough'])
    print '0. Percentage of trips within and between Manhattan and Brooklyn'
    print boro.size()/boro.size().sum()
    print '\n\n\n'
    
    # Question 1
    queryExpr = '(pickup_borough == "Brooklyn" & dropoff_borough == "Brooklyn")'
    brooklyn = yelgreen.query(queryExpr)
    print '1.1 Aggregation efficiency within Brooklyn'
    print nyc.get_reduced_rides(brooklyn)
    print '\n'
    
    queryExpr = '(pickup_borough == "Manhattan" & dropoff_borough == "Brooklyn") |'
    queryExpr += '(pickup_borough == "Brooklyn" & dropoff_borough == "Manhattan")'
    brookman = yelgreen.query(queryExpr)
    print '1.2 Aggregation efficiency between Brooklyn and Manhattan'
    print nyc.get_reduced_rides(brookman)
    print '\n\n\n'
    
    # Question 2
    records = []
    for i in xrange(1,11):
        resol = 0.001*i

        queryExpr = '(pickup_borough == "Brooklyn" & dropoff_borough == "Brooklyn")'
        br = nyc.get_reduced_rides(yelgreen.query(queryExpr),resol=resol)
        queryExpr = '(pickup_neighborhood == "Upper East Side" & dropoff_neighborhood == "Upper East Side")'
        ues = nyc.get_reduced_rides(yelgreen.query(queryExpr),resol=resol)

        records.append((br[0],br[1],br[5],br[6],ues[0],ues[1],ues[5],ues[6]))
        
    cols = ['ntrips_brooklyn_pp','percent_brooklyn_pp','ntrips_brooklyn_pa','percent_brooklyn_pa',
            'ntrips_upperEast_pp','percent_upperEast_pp','ntrips_upperEast_pa','percent_upperEast_pa',]
    ret = pd.DataFrame.from_records(records,columns=cols)
    ret.index = [0.02*i for i in xrange(1,11)]
    
    fig,axarr=plt.subplots(2,sharex=True)
    ret[filter(lambda x: 'ntrips' in x, ret.columns)].plot(marker='o',ax=axarr[0])
    ret[filter(lambda x: 'percent' in x, ret.columns)].plot(marker='o',ax=axarr[1])
    axarr[0].set_ylabel('Number of saved rides')
    axarr[1].set_xlabel('Spatial resolution of pickup location [mile]')
    axarr[1].set_ylabel('Percent of saved rides')
    axarr[0].set_title('pp: point-point aggregation;   pa: point-angle aggregation')
    fig.tight_layout()
    fig.savefig('saveTrips_brookly_upperEast.png')
    
    # Question 3
    queryExpr = '(pickup_neighborhood == "Midtown" & dropoff_neighborhood == "Upper East Side") |'
    queryExpr += '(pickup_neighborhood == "Upper East Side" & dropoff_neighborhood == "Midtown")'
    print '3. Aggregation efficiency between Upper East Side and Midtown'
    print nyc.get_reduced_rides(yelgreen.query(queryExpr))
    print '\n\n\n'
    
    # Questions 4
    # within Brooklyn
    _, fig = nyc.plot_week_hour(brooklyn)
    fig.tight_layout()
    fig.savefig('saveTrips_brookly_week_hour.png')
    
    # Between Brooklyn and Manhattan
    _, fig = nyc.plot_week_hour(brookman)
    fig.tight_layout()
    fig.savefig('saveTrips_brookly_manhattan_week_hour.png')
    
    # scatter plots to trace out the borough shape
    fig,ax=plt.subplots()
    yelgreen.plot(kind='scatter',x='pickup_longitude',y='pickup_latitude',s=0.02,c='k',ax=ax, edgecolors='none')
    ax.axis('off')
    ax.set_title('Pickups')
    fig.tight_layout()
    fig.savefig('brookman_pickup.png')
    
    fig,ax=plt.subplots()
    yelgreen.plot(kind='scatter',x='dropoff_longitude',y='dropoff_latitude',s=0.02,c='k',ax=ax, edgecolors='none')
    ax.axis('off')
    ax.set_title('Dropoffs')
    fig.tight_layout()
    fig.savefig('brookman_dropoff.png')
    
    # for fun, top 10 pickups and dropoffs spots
    fig,_ = nyc.plot_top_pickups_dropoffs(yelgreen,10)
    fig.tight_layout()
    fig.savefig('top_n_spots.png')
               
    return 


# run `python goBooklyn.py ` to excute the following code lines.
if __name__ == '__main__':
    
    #---This following two lines is to make the first clean of the raw csv file from TCL website.
    #---Since it is expensive, save the result as csv file for further EDA analysis. 
    # transform_nyc_taxi_file('data/green_tripdata_2016-06.csv')
    # transform_nyc_taxi_file('data/yellow_tripdata_2016-06.csv')
    
    #---brooklyn expansion analysis
    brooklyn_expansion_analysis()
