import json
import pandas as pd
import numpy as np

def load_data(file_list, df_activities, df_links_network):
    data_frames = []
    for file in file_list:
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data['link_counts'], dict):
                data['link_counts'] = data['link_counts'].values()
            df_links = pd.DataFrame({
                'link_id': data['links_id'],
                'link_from': data['link_from'],
                'link_to': data['link_to'],
                'link_length': data['link_length'],
                'link_freespeed': data['link_freespeed'],
                'link_capacity': data['link_capacity'],
                'link_permlanes': data['link_permlanes'],
                'link_counts': data['link_counts']
            })
            df_nodes = pd.DataFrame({
                'node_id': data['nodes_id'],
                'node_x': data['nodes_x'],
                'node_y': data['nodes_y']
            })
            df_od_pairs = pd.DataFrame(data['o_d_pairs'], columns=['origin', 'destination'])
            
            df_work = pd.DataFrame({
                        'work_x': data['work_x'],
                        'work_y': data['work_y'],
                        'go_to_work': data['go_to_work']
            })
            df_home = pd.DataFrame({
                'home_x': data['home_x'],
                'home_y': data['home_y'],
                'go_to_home': data['go_to_home']
            })
            
            df_links = df_links.merge(df_nodes, how='left', left_on='link_from', right_on='node_id')
            df_links = df_links.rename(columns={'node_x': 'start_node_x', 'node_y': 'start_node_y'})
            df_links.drop('node_id', axis=1, inplace=True)
            df_links = df_links.merge(df_nodes, how='left', left_on='link_to', right_on='node_id')
            df_links = df_links.rename(columns={'node_x': 'end_node_x', 'node_y': 'end_node_y'})
            df_links.drop('node_id', axis=1, inplace=True) 
            
            origin_counts = df_od_pairs['origin'].value_counts()
            df_origin_counts = origin_counts.reset_index()
            df_origin_counts.columns = ['origin', 'start_count']
            destination_counts = df_od_pairs['destination'].value_counts()
            df_destination_counts = destination_counts.reset_index()
            df_destination_counts.columns = ['destination', 'end_count']
            df_links = df_links.merge(df_origin_counts, how='left', left_on='link_from', right_on='origin')
            df_links.drop('origin', axis=1, inplace=True)
            df_links = df_links.merge(df_destination_counts, how='left', left_on='link_to', right_on='destination')
            df_links.drop('destination', axis=1, inplace=True)
            df_links[['start_count','end_count']] = df_links[['start_count','end_count']].fillna(-1)
            
            # Calculate time of go_to_work and go_to_sum
            df_act_work = df_activities[df_activities['activity_type_main']=='work'].drop(['end_time'], axis=1)
            df_act_work = df_act_work.merge(df_work, how='left', left_on=['x','y'], right_on=['work_x','work_y'])
            df_act_work.drop(['x','y'], axis=1, inplace=True)
            df_act_work_agg = df_act_work.groupby(by="link")['go_to_work'].sum().reset_index(drop=False)
            df_act_home = df_activities[df_activities['activity_type_main']=='home'].drop(['end_time'], axis=1)
            df_act_home = df_act_home.merge(df_home, how='left', left_on=['x','y'], right_on=['home_x','home_y'])
            df_act_home.drop(['x','y'], axis=1, inplace=True)
            df_act_home_agg = df_act_home.groupby(by="link")['go_to_home'].sum().reset_index(drop=False)
            df_act_agg = df_act_home_agg.merge(df_act_work_agg, how='outer', on='link')
            df_act_agg.fillna(0, inplace=True)
            df_act_agg['go_to_sum'] = df_act_agg['go_to_home'] + df_act_agg['go_to_work']

            df_rushhr = df_activities[df_activities['end_time']!=-1]
            df_rushhr.loc[:, 'rush_hour'] = 0
            df_rushhr.loc[df_rushhr['end_time'].between(pd.to_timedelta('08:00:00'), pd.to_timedelta('10:00:00'), inclusive='both'), 'rush_hour'] = 1
            df_rushhr.loc[df_rushhr['end_time'].between(pd.to_timedelta('16:00:00'), pd.to_timedelta('19:00:00'), inclusive='both'), 'rush_hour'] = 1
            df_rushhr = df_rushhr[['link', 'rush_hour']]
            df_rushhragg = df_rushhr.groupby(by="link").sum()['rush_hour'].reset_index(drop=False)
            
            df_maxduragg = df_activities[df_activities['max_dur']!=-1].groupby(by='link')['max_dur'].sum().reset_index(drop=False)
            
            df_activities['cemdapStopDuration_s'] = df_activities['cemdapStopDuration_s'].astype(float)
            df_cemagg = df_activities[df_activities['cemdapStopDuration_s']!=-1].groupby(by='link')['cemdapStopDuration_s'].sum().reset_index(drop=False)
            
            df_activities['income'] = df_activities['income'].astype(float)
            df_income = df_activities[df_activities['income']!=-1].groupby(by='link')['income'].sum().reset_index(drop=False)
            df_income_avg = df_activities[df_activities['income']!=-1].groupby(by='link')['income'].mean().reset_index(drop=False)
            df_income_avg.columns = ['link', 'income_avg']
            
            df_activities['score'] = df_activities['score'].astype(float)
            df_score = df_activities[df_activities['score']!=-1].groupby(by='link')['score'].sum().reset_index(drop=False)
            df_score_avg = df_activities[df_activities['score']!=-1].groupby(by='link')['score'].mean().reset_index(drop=False)
            df_score_avg.columns = ['link', 'score_avg']
            
            df_zone = df_activities[df_activities['home-activity-zone']!=-1].groupby(by='link')['home-activity-zone'].agg(lambda x: '_'.join(x.unique())).reset_index(drop=False)
            df_zone.columns = ['link', 'home-activity-zone']

            df_temp = df_links.merge(df_links_network, how='left', on=['start_node_x','start_node_y','end_node_x','end_node_y'])
            df_temp = df_temp[['link_id_x','link_from','link_to','link_id_y','from', 'to', 'type']]
            
            df_temp = df_temp.merge(df_income, how='left', left_on='link_id_y', right_on='link')
            df_temp.drop('link', axis=1, inplace=True)
            df_temp = df_temp.merge(df_income_avg, how='left', left_on='link_id_y', right_on='link')
            df_temp.drop('link', axis=1, inplace=True)
            df_temp = df_temp.merge(df_score, how='left', left_on='link_id_y', right_on='link')
            df_temp.drop('link', axis=1, inplace=True)    
            df_temp = df_temp.merge(df_score_avg, how='left', left_on='link_id_y', right_on='link')
            df_temp.drop('link', axis=1, inplace=True)  
            df_temp = df_temp.merge(df_zone, how='left', left_on='link_id_y', right_on='link')
            df_temp.drop('link', axis=1, inplace=True)    
            
            df_temp = df_temp.merge(df_act_agg, how='left', left_on='link_id_y', right_on='link')
            df_temp.drop('link', axis=1, inplace=True)
            df_temp = df_temp.merge(df_rushhragg, how='left', left_on='link_id_y', right_on='link')
            df_temp.drop('link', axis=1, inplace=True)
            df_temp = df_temp.merge(df_maxduragg, how='left', left_on='link_id_y', right_on='link')
            df_temp.drop('link', axis=1, inplace=True)
            df_temp = df_temp.merge(df_cemagg, how='left', left_on='link_id_y', right_on='link')
            df_temp.fillna({'cemdapStopDuration_s':-1, 'max_dur':-1, 'rush_hour': -1, 'go_to_sum': -1, 'income': -1, 'income_avg': -1, 'score': -1, 'score_avg': -1, 'home-activity-zone': 'default'}, inplace=True)
            
            df_temp = df_temp[['link_id_x', 'go_to_sum', 'income','income_avg', 'score', 'score_avg', 'home-activity-zone', 'rush_hour', 'max_dur', 'cemdapStopDuration_s', 'type']]

            df_links = df_links.merge(df_temp, how='left', left_on='link_id', right_on='link_id_x')
            df_links.drop('link_id_x', axis=1, inplace=True)
            df_links['length_per_capacity_ratio'] = df_links['link_length'] / df_links['link_capacity']
            df_links['speed_capacity_ratio'] = df_links['link_freespeed'] / df_links['link_capacity']
            df_links['length_times_lanes'] = df_links['link_length'] * df_links['link_permlanes']
            df_links['speed_times_capacity'] = df_links['link_freespeed'] * df_links['link_capacity']
            df_links['link_times'] = df_links['link_length'] / df_links['link_freespeed']
            df_links['capacity_divided_by_lanes'] = df_links['link_capacity'] / df_links['link_permlanes']
            

        data_frames.append(df_links)
    return pd.concat(data_frames, ignore_index=True)
