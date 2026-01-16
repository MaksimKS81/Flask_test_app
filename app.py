"""

versions: 1.0.3 added time selection for subject and cohort

"""





from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv  # load local .env in development
load_dotenv()
import pandas as pd
import os
import json
from functools import wraps
from flask import jsonify, abort


GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("Missing SECRET_KEY: set SECRET_KEY in .env file")



#---------------------------- constants -------------------------------
unilateral_squat_test_all = ( ['summary__unilateral_squat__mobility__ankle__left',
                          'summary__unilateral_squat__mobility__ankle__right'],
                            ['summary__unilateral_squat__mobility__hip__left',
                          'summary__unilateral_squat__mobility__hip__right'],
                            ['summary__unilateral_squat__mobility__knee__left',
                          'summary__unilateral_squat__mobility__knee__right'],
                            ['summary__unilateral_squat__mobility__depth__left',
                          'summary__unilateral_squat__mobility__depth__right'],
                            ['summary__unilateral_squat__alignment__dyn_valgus__left',
                          'summary__unilateral_squat__alignment__dyn_valgus__right'],
                            )

bilateral_squat_all = ( ['summary__bilateral_squat__mobility__ankle__left',
                          'summary__bilateral_squat__mobility__ankle__right'],
                        ['summary__bilateral_squat__mobility__hip__left',
                          'summary__bilateral_squat__mobility__hip__right'],
                        ['summary__bilateral_squat__mobility__knee__left',
                          'summary__bilateral_squat__mobility__knee__right'],
                        ['summary__bilateral_squat__alignment__dyn_valgus__left',
                          'summary__bilateral_squat__alignment__dyn_valgus__right'],
                        ['summary__bilateral_squat__alignment__fem_rot__left',
                         'summary__bilateral_squat__alignment__fem_rot__right']
                      )
lateral_launch_all = ( ['summary__lateral_launch__mobility__ankle__left',
                          'summary__lateral_launch__mobility__ankle__right'],
                        ['summary__lateral_launch__mobility__hip__left',
                          'summary__lateral_launch__mobility__hip__right'],
                        ['summary__lateral_launch__mobility__knee__left',
                          'summary__lateral_launch__mobility__knee__right'],
                        ['summary__lateral_launch__alignment__dyn_valgus__left',
                          'summary__lateral_launch__alignment__dyn_valgus__right'],
                        ['summary__lateral_lunge__mobility__stride__left',
                        'summary__lateral_lunge__mobility__stride__right'],
                      )
stork_stance_all = ( ['summary__stork_stance__mobility__ankle__left',
                          'summary__stork_stance__mobility__ankle__right'],
                        ['summary__stork_stance__mobility__hip__left',
                          'summary__stork_stance__mobility__hip__right'],
                        ['summary__stork_stance__mobility__knee__left',
                          'summary__stork_stance__mobility__knee__right'],
                        ['summary__stork_stance__alignment__dyn_valgus__left',
                          'summary__stork_stance__alignment__dyn_valgus__right'],
                        ['summary__stork_stance__mobility__COM__left',
                        'summary__stork_stance__mobility__COM__right']
                      )
step_down_all = ( ['summary__step_down__mobility__ankle__left',
                          'summary__step_down__mobility__ankle__right'],
                        ['summary__step_down__mobility__hip__left',
                          'summary__step_down__mobility__hip__right'],
                        ['summary__step_down__mobility__knee__left',
                          'summary__step_down__mobility__knee__right'],
                        ['summary__step_down__alignment__dyn_valgus__left',
                          'summary__step_down__alignment__dyn_valgus__right'],
                        ['summary__step_down__mobility__rate__left',
                        'summary__step_down__mobility__rate__right']
                      )
front_lunge_all = ( ['summary__front_lunge__mobility__ankle__left',
                          'summary__front_lunge__mobility__ankle__right'],
                        ['summary__front_lunge__mobility__hip__left',
                          'summary__front_lunge__mobility__hip__right'],
                        ['summary__front_lunge__mobility__knee__left',
                          'summary__front_lunge__mobility__knee__right'],
                        ['summary__front_lunge__alignment__dyn_valgus__left',
                          'summary__front_lunge__alignment__dyn_valgus__right'],
                        ['summary__front_lunge__mobility__trail_hip_ext__left',
                        'summary__front_lunge__mobility__trail_hip_ext__right']
                      )
#-------------------------- local functions ----------------------------
def stat_calc(data_ps, filter = [0.01, 0.99]):
    # Calculate 1st and 99th percentiles for filtering
    lower_bound = data_ps.quantile(filter[0])
    upper_bound = data_ps.quantile(filter[1])
    
    # Filter data based on these percentiles
    filtered = data_ps[(data_ps >= lower_bound) & 
                     (data_ps <= upper_bound)]
    
    median_1 = round(filtered.quantile(0.50), 3)
    
    max_1 = filtered.max()
    
    min_1 = filtered.min()
    
    # Calculate quartiles for confidence interval
    q25 = filtered.quantile(0.25)
    q75 = filtered.quantile(0.75)
    
    return {'max': max_1, 'median': median_1, 'min': min_1, 'q25': q25, 'q75': q75}

#-------------------------- Flask app setup ----------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
#------------------------------------------------------------------------

def percent_diff(a, b):
    """
    Calculate the percentage difference between two values.
    Returns a string formatted as a percentage with 2 decimal places.
    """
    if a == 0 and b == 0:
        return 0.0
    elif a == 0 or b == 0:
        return 100.0
    else:
        diff = abs(a - b)
        avg = (a + b) / 2
        percent_diff = (diff / avg) * 100
        return round(percent_diff, 1)

def create_bar_chart_from_polar(polar_traces, categories, single_uid, normed_data_uid, uid, uid_list, time_subject, type_of_test, uid_inj, width, height, display_mode='absolute', all_info=None):
    """
    Convert polar/radar chart data to a grouped bar chart format.
    Shows normalized values for each metric comparing Right vs Left sides.
    """
    # Categories should already be clean (no duplicate), but double-check
    if len(categories) > 0 and categories[-1] == categories[0]:
        clean_categories = categories[:-1]
    else:
        clean_categories = categories
    
    # Extract the base metric names (without the HTML formatting and values)
    metric_names = []
    for cat in clean_categories:
        # Extract just the metric name before <br>
        base_name = cat.split('<br>')[0] if '<br>' in cat else cat
        metric_names.append(base_name)
    
    # Get normalized values - normed_data_uid has been extended with duplicate, so remove it
    right_values = normed_data_uid.get('Right', [])
    left_values = normed_data_uid.get('Left', [])
    
    # Debug: print lengths
    print(f"[DEBUG BAR_CHART] Categories: {len(metric_names)}, Right: {len(right_values)}, Left: {len(left_values)}")
    
    # Remove closing duplicate if present
    if len(right_values) > len(metric_names):
        right_values = right_values[:-1]
    if len(left_values) > len(metric_names):
        left_values = left_values[:-1]
    
    print(f"[DEBUG BAR_CHART] After trim - Right: {len(right_values)}, Left: {len(left_values)}")
    print(f"[DEBUG BAR_CHART] Right values: {right_values}")
    print(f"[DEBUG BAR_CHART] Left values: {left_values}")
    
    # Prepare text annotations based on display mode
    right_text = []
    left_text = []
    
    if all_info is not None:
        for idx, k in enumerate(all_info.keys()):
            # Get absolute values
            right_val = round(single_uid['Right'][idx], 2)
            left_val = round(single_uid['Left'][idx], 2)
            
            if display_mode == 'zscore':
                # Calculate z-score with 0 at minimum value
                lo = all_info[k]['good']['min']
                hi = all_info[k]['good']['max']
                median_val = all_info[k]['good']['median']
                std_dev = (hi - lo) / 4 if (hi - lo) != 0 else 1
                z_right = round((right_val - lo) / std_dev, 2)
                z_left = round((left_val - lo) / std_dev, 2)
                right_text.append(f"{z_right}")
                left_text.append(f"{z_left}")
            else:  # absolute mode
                right_text.append(f"{right_val}")
                left_text.append(f"{left_val}")
    else:
        # Fallback: show normalized values
        right_text = [f"{round(v, 2)}" for v in right_values]
        left_text = [f"{round(v, 2)}" for v in left_values]
    
    # Determine colors based on injured side
    if uid_inj == 'Right':
        right_color = "red"
        left_color = "lightblue"
        right_label = "Right Side (Injured)"
        left_label = "Left Side"
    elif uid_inj == 'Left':
        right_color = "lightblue"
        left_color = "red"
        right_label = "Right Side"
        left_label = "Left Side (Injured)"
    else:
        right_color = "lightblue"
        left_color = "lightcoral"
        right_label = "Right Side"
        left_label = "Left Side"
    
    # Create bar chart traces with text annotations
    traces = [
        {
            "type": "bar",
            "x": metric_names,
            "y": right_values,
            "name": right_label,
            "marker": {"color": right_color},
            "text": right_text,
            "textposition": "outside",
            "textfont": {"size": 10}
        },
        {
            "type": "bar",
            "x": metric_names,
            "y": left_values,
            "name": left_label,
            "marker": {"color": left_color},
            "text": left_text,
            "textposition": "outside",
            "textfont": {"size": 10}
        }
    ]
    
    layout = {
        "barmode": "group",
        "width": width,
        "height": height,
        "title": f"Bar Chart: {type_of_test} - Subject #{uid_list.index(uid)} at {time_subject}",
        "xaxis": {
            "title": "Metrics",
            "tickangle": -45
        },
        "yaxis": {
            "title": "Normalized Value (0-1)",
            "range": [0, 1.1]
        },
        "margin": {
            "b": 150
        },
        "showlegend": True,
        "shapes": [
            # Red zone (0.0 - 0.3)
            {
                "type": "rect",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "y0": 0.0,
                "x1": 1,
                "y1": 0.3,
                "fillcolor": "rgba(255, 0, 0, 0.2)",
                "line": {"width": 0},
                "layer": "below"
            },
            # Yellow zone (0.3 - 0.6)
            {
                "type": "rect",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "y0": 0.3,
                "x1": 1,
                "y1": 0.6,
                "fillcolor": "rgba(255, 255, 0, 0.2)",
                "line": {"width": 0},
                "layer": "below"
            },
            # Green zone (0.6 - 1.0)
            {
                "type": "rect",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "y0": 0.6,
                "x1": 1,
                "y1": 1.0,
                "fillcolor": "rgba(0, 255, 0, 0.2)",
                "line": {"width": 0},
                "layer": "below"
            }
        ]
    }
    
    return {"traces": traces, "layout": layout}

def create_bar_chart_from_polar_multi_time(polar_traces, all_times, single_uid, normed_data_uid, uid, uid_list, time_subject, type_of_test, uid_inj, width, height):
    """
    Convert polar/radar chart data with multiple timepoints to a grouped bar chart format.
    Shows normalized values for each metric across different time periods.
    """
    # Extract metric names and data from the polar traces
    metric_names = []
    time_data = {}  # {time_period: [values]}
    
    # Parse traces to extract data
    for trace in polar_traces:
        if 'theta' in trace and 'r' in trace:
            trace_name = trace.get('name', '')
            
            # Skip fill traces and non-data traces
            if 'fill' in trace or 'showlegend' in trace and not trace['showlegend']:
                continue
            
            # Extract time period from trace name (e.g., "Injured Side - 3 Month")
            if ' - ' in trace_name:
                parts = trace_name.split(' - ')
                if len(parts) >= 2:
                    time_period = parts[-1]
                    
                    # Get theta values (metric names) - remove duplicates
                    theta_values = trace['theta']
                    r_values = trace['r']
                    
                    # Remove closing duplicate if present
                    if len(theta_values) > 0 and theta_values[0] == theta_values[-1]:
                        theta_values = theta_values[:-1]
                        r_values = r_values[:-1]
                    
                    # Store data
                    if not metric_names and theta_values:
                        metric_names = [t.split('<br>')[0] if '<br>' in t else t for t in theta_values]
                    
                    time_data[time_period] = r_values
    
    # If we couldn't extract from traces, use the single_uid data
    if not metric_names and single_uid:
        metric_names = list(range(len(single_uid.get('Right', []))))
    
    # Create bar chart traces for each time period
    traces = []
    colors = {
        '3 Month': 'lightblue',
        '6 Month': 'lightgreen', 
        '9 Month': 'lightsalmon',
        '12 Month': 'lightpink'
    }
    
    for time_period, values in time_data.items():
        trace = {
            "type": "bar",
            "x": metric_names,
            "y": values,
            "name": time_period,
            "marker": {"color": colors.get(time_period, "gray")}
        }
        traces.append(trace)
    
    layout = {
        "barmode": "group",
        "width": width,
        "height": height,
        "title": f"Bar Chart: {type_of_test} - Subject #{uid_list.index(uid)} - Injured Side Progress",
        "xaxis": {
            "title": "Metrics",
            "tickangle": -45
        },
        "yaxis": {
            "title": "Normalized Value",
            "range": [0, 1.1]
        },
        "showlegend": True
    }
    
    return {"traces": traces, "layout": layout}

    



script_dir = os.path.dirname(__file__)
pkl_path = os.path.join(script_dir, 'all_my_data.pkl')

try:
    df_test = pd.read_pickle(pkl_path)
    print(f"Loaded {pkl_path}")
except Exception as e:
    print(f"Failed to load {pkl_path}: {e}")
    df_test = pd.DataFrame()  # fallback empty df


uid_counts = df_test['uid'].value_counts().reset_index()
uid_counts.columns = ['uid', 'count']

uid_list = list(uid_counts.iloc[:10]['uid']) # first 10 

time_points = ['3 Month','6 Month','9 Month','12 Month']

if 'test' in uid_list:
    uid_list.remove('test')  # Remove 'test' if it exists

""" client_secret_path = os.path.join(script_dir, 'client_secret_564818052534-f8f29ltrvhh7kmcn7i7omadbrd4cg3e0.apps.googleusercontent.com.json')
with open(client_secret_path, 'r') as f:
  client_secret_data = json.load(f)

GOOGLE_CLIENT_ID = client_secret_data['web']['client_id']
GOOGLE_CLIENT_SECRET = client_secret_data['web']['client_secret'] """


app = Flask(__name__)

# Set a secret key for the Flask app
app.secret_key = 'your_secret_key'  # Set a unique and secret key for session management

# Initialize OAuth and register the Google client
oauth = OAuth(app)
google = oauth.register(
     name='google',
     client_id=GOOGLE_CLIENT_ID,
     client_secret=GOOGLE_CLIENT_SECRET,
     server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
     api_base_url='https://openidconnect.googleapis.com/v1/',
     client_kwargs={'scope': 'openid email profile'},
)

def func_plot(uid,
                time_subject, 
                time_cohort, 
                temp_ch_list,
                type_of_test = 'unilateral_squat', 
                ch_type = 'mobility',
                width=500, 
                height=500):
#-------- data preprocessing ----------------    

    global uid_list

    time = '053-Time after Surgery'
    gender = '005-Gender'
    injure = 'injured_side'
    global time_points 

    

    #uid = uid_list[0]
    #temp_ch_list = unilateral_squat_test_all
    #time_v = '9 Month'
    all_info = {}
    for i in temp_ch_list:
    
        temp = i[0].split('__')
        if 'pct' in i[0]:
            axis = temp[2] + '__' + temp[3] + '_pct'
        else:
            axis = temp[2] + '__' + temp[3]
    
           
        y_temp_spider = df_test[[time, *i, injure]]
        y_temp = y_temp_spider[y_temp_spider[time] == time_cohort]
        
        good_values = y_temp[y_temp[injure] == 'Left'][i[1]].values.tolist() + y_temp[y_temp[injure] == 'Right'][i[0]].values.tolist()
        good_values_ps = pd.Series(good_values)
        
        bad_values = y_temp[y_temp[injure] == 'Left'][i[0]].values.tolist() + y_temp[y_temp[injure] == 'Right'][i[1]].values.tolist()
        bad_values_ps = pd.Series(bad_values)
        all_info.update({axis:{'keys': i, 'bad':stat_calc(bad_values_ps), 'good':stat_calc(good_values_ps)}})


    
    ranges = []
    for k in all_info:
    
        aaa = [max([all_info[k]['good']['max'],all_info[k]['bad']['max']]), min(all_info[k]['bad']['min'], all_info[k]['good']['min'])]
        aaa.sort()
    
        ranges.append([*aaa, k])  # k is the channel name (axis)
    #print(ranges)
    
    median_good = []
    for k in all_info:
        median_good.append(all_info[k]['good']['median'])
    
    max_good = []
    for k in all_info:
        max_good.append(all_info[k]['good']['max'])

    max_bad = []
    for k in all_info:
        max_bad.append(all_info[k]['bad']['max'])

    min_bad = []
    for k in all_info:
        min_bad.append(all_info[k]['bad']['min'])
    
    median_bad = []
    for k in all_info:
        median_bad.append(all_info[k]['bad']['median'])
    
    #categories = [k for k in all_info]

        #--------------------------------- single UID -------------------------
    
    
    single_uid = {'Right':[], 'Left':[] }
    for ch in temp_ch_list:
        y_temp_spider = df_test[[time, *ch, injure, 'uid']]
        y_temp = y_temp_spider[y_temp_spider[time] == time_subject]
        
        # Debugging output to log the state of variables
#        print("Processing channel:", ch)
#        print("Filtered DataFrame for time_v:", y_temp)
 #       print("UID being processed:", uid)

        # Safely access the first element of the list, or assign a default value if the list is empty
        right_values = list(y_temp[y_temp['uid'] == uid][ch[1]])
        if not right_values:
            right_values = [0]
        left_values = list(y_temp[y_temp['uid'] == uid][ch[0]])
        if not left_values:
            left_values = [0]

#        print("Right values:", right_values)
#        print("Left values:", left_values)

        single_uid['Right'].append(right_values[0] if right_values else None)
        single_uid['Left'].append(left_values[0] if left_values else None)


    print(f"[DEBUG] single_uid raw values plot_1: {single_uid}")
    
    uid_inj_list = list(y_temp[y_temp['uid'] == uid][injure])
    uid_inj = uid_inj_list[0] if uid_inj_list else 'NO RECORDS'
    
 #   print('Injured:' + uid_inj)
    
    normed_data_uid = {}
    
    for label, vals in single_uid.items():
        # vals: e.g. [15, 80, 40, 7, 150, 25]
        norm = []
        for v, (lo, hi, ch_name) in zip(vals, ranges):
            # Normalize by the range of the uninjured side
            if 'alignment' in ch_name:
                norm.append(1 - ((v - lo) / (hi - lo)))
            else:
                norm.append((v - lo) / (hi - lo))
        normed_data_uid[label] = norm
    # close the loop for each UID trace so the radar line connects back
    for label in normed_data_uid:
        normed_data_uid[label].append(normed_data_uid[label][0])

    categories = [k for k in all_info]

    
    #ranges = range
    #print(categories)
    #print(range)
    
    values_dict = {'50th bad':median_bad, '50th good': median_good, 'max good': max_good, 'max bad': max_bad, 'min bad': min_bad}
    #print(values_dict)

    
    # Normalize values per axis and close the loop by repeating first point
    normed_data = {}
    for label, vals in values_dict.items():
        # vals: e.g. [15, 80, 40, 7, 150, 25]
        norm = []
        for v, (lo, hi, ch_name) in zip(vals, ranges):

            norm.append((v - lo) / (hi - lo))

        norm.append(norm[0])
        normed_data[label] = norm
    
    #
    
    # Save original categories before closing the loop for bar chart
    original_categories = categories.copy()
    
    # Prepare theta (angles) and extend categories to close the loop
    theta = categories + [categories[0]]

#--------------------------------------------------------------------    

    categories = categories + [categories[0]]
    traces = []

    for label, norm_vals in normed_data_uid.items():
        # Marker colors based on normalized value
        marker_colors = norm_vals[:-1]  # exclude duplicate
        # Text annotations are actual values
        text_vals = [str(v) for v in single_uid[label]] + [str(single_uid[label][0])]
        trace = {
                "type": "scatterpolar",
                "r": norm_vals,
                "theta": theta,
                "name": f"Test subject #{uid_list.index(uid)} {label}, side",
                "text": text_vals,
                "textposition": "top center"
        }
        traces.append(trace)   
    # for i, values in enumerate(values_list):
    #     values = values + [values[0]]
    #     trace = {
    #         "type": "scatterpolar",
    #         "r": values,
    #         "theta": categories,
    #         "fill": "toself",
    #         "name": f"Trace {i+1}"
    #     }
    #     traces.append(trace)

    max_good_norm = list(normed_data['max good'])
    max_good_norm += [max_good_norm[0]]

    max_bad_norm = list(normed_data['max bad'])
    max_bad_norm += [max_bad_norm[0]]
    
    
    median_good_norm = list(normed_data['50th good'])
    median_good_norm += [median_good_norm[0]]

    median_bad_norm = list(normed_data['50th bad'])
    median_bad_norm += [median_bad_norm[0]]
    
    
    min_bad_norm = list(normed_data['min bad'])
    min_bad_norm += [min_bad_norm[0]]
    if uid_inj != 'NO RECORDS':
        trace = {
                "type": "scatterpolar",
                "r": median_bad_norm,
                "theta": theta,
                "mode": 'lines',
                "line": dict(color='rgba(0,0,0,0)'),
                "showlegend": False
        }
        traces.append(trace)      

        trace = {
                "type": "scatterpolar",
                "r": min_bad_norm,
                "theta": theta,
                "mode": 'lines',
                "fill": 'tonext',
                "fillcolor": 'rgba(255,0,0,0.25)',
                "line": dict(color='rgba(0,0,0,0)'),
                "name":"Injured level below 50th percentile"
        }
        traces.append(trace)     

        trace = {
                "type": "scatterpolar",
                "r": max_good_norm,
                "theta": theta,
                "mode": 'lines',
                "line": dict(color='rgba(0,0,0,0)'),
                "showlegend": False
        }
        traces.append(trace)      

        trace = {
                "type": "scatterpolar",
                "r": median_good_norm,
                "theta": theta,
                "mode": 'lines',
                "fill": 'tonext',
                "fillcolor": 'rgba(0,255,0,0.4)',
                "line": dict(color='rgba(0,0,0,0)'),
                "name":"Healthy level above 50th percentile"
        }
        traces.append(trace)    


    layout = {
        "polar": {
            "radialaxis": {
                "visible": True,
                "range": [0, 1]
            }
        },
        "width": width,
        "height": height,
        "showlegend": True,
        "showticklabels": False,
        "title": f"Exe. type: {type_of_test}, subject: {time_subject}, Injured side: {uid_inj}"
    }
    
    # Create bar chart version with the same data
    bar_chart = create_bar_chart_from_polar(traces, original_categories, single_uid, normed_data_uid, uid, uid_list, time_subject, type_of_test, uid_inj, width, height, display_mode='absolute', all_info=all_info)
    
    return {"traces": traces, "layout": layout, "bar_chart": bar_chart}

def func_plot_2(uid,    
                time_subject, 
                time_cohort, 
                temp_ch_list,
                type_of_test = 'unilateral_squat', 
                ch_type = 'mobility',
                width=500, 
                height=500,
                display_mode='absolute'):
#-------- data preprocessing ----------------    

    global uid_list

    time = '053-Time after Surgery'
    gender = '005-Gender'
    injure = 'injured_side'
    global time_points 

    # Initialize variables
    single_uid = {'Right': [], 'Left': []}
    normed_data_uid = {}
    normed_data = {}
    uid_inj = 'NO RECORDS'

    #uid = uid_list[0]
    #temp_ch_list = unilateral_squat_test_all
    #time_v = '9 Month'
    all_info = {}
    for i in temp_ch_list:
    
        temp = i[0].split('__')
        if 'pct' in i[0]:
            axis = temp[2] + '__' + temp[3] + '_pct'
        else:
            axis = temp[2] + '__' + temp[3]
    
           
        y_temp_spider = df_test[[time, *i, injure]]
        y_temp = y_temp_spider[y_temp_spider[time] == time_cohort]
        
        good_values = y_temp[y_temp[injure] == 'Left'][i[1]].values.tolist() + y_temp[y_temp[injure] == 'Right'][i[0]].values.tolist()
        good_values_ps = pd.Series(good_values)
        
        bad_values = y_temp[y_temp[injure] == 'Left'][i[0]].values.tolist() + y_temp[y_temp[injure] == 'Right'][i[1]].values.tolist()
        bad_values_ps = pd.Series(bad_values)
        all_info.update({axis:{'keys': i, 'bad':stat_calc(bad_values_ps), 'good':stat_calc(good_values_ps)}})


    
    ranges = []
    for k in all_info:
    
        aaa = [max([all_info[k]['good']['max'],all_info[k]['bad']['max']]), min(all_info[k]['bad']['min'], all_info[k]['good']['min'])]
        aaa.sort()
    
        ranges.append([*aaa, k])  # k is the channel name (axis)

    print(f"[DEBUG] axis range values plot_2: {ranges}")

    median_good = []
    for k in all_info:
        median_good.append(all_info[k]['good']['median'])
    
    max_good = []
    for k in all_info:
        max_good.append(all_info[k]['good']['max'])

    max_bad = []
    for k in all_info:
        max_bad.append(all_info[k]['bad']['max'])

    min_bad = []
    for k in all_info:
        min_bad.append(all_info[k]['bad']['min'])
    
    median_bad = []
    for k in all_info:
        median_bad.append(all_info[k]['bad']['median'])
    
    #categories = [k for k in all_info]

        #--------------------------------- single UID -------------------------
    
    
    single_uid = {'Right':[], 'Left':[] }
    for ch in temp_ch_list:
        y_temp_spider = df_test[[time, *ch, injure, 'uid']]
        y_temp = y_temp_spider[y_temp_spider[time] == time_subject]
        
        # Debugging output to log the state of variables
#        print("Processing channel:", ch)
#        print("Filtered DataFrame for time_v:", y_temp)
 #       print("UID being processed:", uid)

        # Safely access the first element of the list, or assign a default value if the list is empty
        right_values = list(y_temp[y_temp['uid'] == uid][ch[1]])
        if not right_values:
            right_values = [0]
        left_values = list(y_temp[y_temp['uid'] == uid][ch[0]])
        if not left_values:
            left_values = [0]

#        print("Right values:", right_values)
#        print("Left values:", left_values)

        single_uid['Right'].append(right_values[0] if right_values else None)
        single_uid['Left'].append(left_values[0] if left_values else None)
    
    print(f"[DEBUG] single_uid raw values plot_2: {single_uid}")

    uid_inj_list = list(y_temp[y_temp['uid'] == uid][injure])
    uid_inj = uid_inj_list[0] if uid_inj_list else 'NO RECORDS'
    
 #   print('Injured:' + uid_inj)
    
    normed_data_uid = {}
#    print(label)
    
    for label, vals in single_uid.items():
        # vals: e.g. [15, 80, 40, 7, 150, 25]
        norm = []
        for v, (lo, hi, ch_name) in zip(vals, ranges):
            # Normalize by the range of the uninjured side
            if 'alignment' in ch_name:
                norm.append(1 - ((v - lo) / (hi - lo)))
            else:
                norm.append((v - lo) / (hi - lo))
        normed_data_uid[label] = norm
    # close the loop for each UID trace so the radar line connects back
    for label in normed_data_uid:
        normed_data_uid[label].append(normed_data_uid[label][0])

    #categories = [k for k in all_info]
    categories = []
    for idx, k in enumerate(all_info):
        # Get real values for Right, Left, and their absolute difference
        right_val = round(single_uid['Right'][idx], 2)
        left_val = round(single_uid['Left'][idx], 2)

        # Extract the range (lo, hi) for the current axis
        lo = all_info[k]['good']['min']
        hi = all_info[k]['good']['max']
        median_val = all_info[k]['good']['median']
        
        # Calculate z-score with 0 at minimum value: (value - min) / std_dev
        # Approximate std_dev using range: (max - min) / 4 (rough estimate)
        std_dev = (hi - lo) / 4 if (hi - lo) != 0 else 1
        z_right = round((right_val - lo) / std_dev, 2)
        z_left = round((left_val - lo) / std_dev, 2)

        # Calculate the percentile for right_val and left_val
        norm_right = normed_data_uid['Right'][idx]
        norm_left = normed_data_uid['Left'][idx]


        abs_diff = percent_diff(right_val, left_val)


        color = "red" if abs_diff > 10 else "green"
        if norm_right < 0.3:
            color_right = "red"
        elif 0.3 <= norm_right < 0.6:
            color_right = "orange"
        else:
            color_right = "green"

        if norm_left < 0.3:
            color_left = "red"
        elif 0.3 <= norm_left < 0.6:
            color_left = "orange"
        else:
            color_left = "green"
#        print(abs_diff, color)

        # Format the axis name with additional information based on display mode
        if display_mode == 'zscore':
            formatted_name = (
                            f"{k}<br>"
                            f"<span style='color:{color_right};'>R: {z_right}</span>, "
                            f"<span style='color:{color_left};'>L: {z_left}</span>, "
                            f"<span style='color:{color};'>Δ: {abs_diff}%</span>"
                            )
        else:  # absolute mode
            formatted_name = (
                            f"{k}<br>"
                            f"<span style='color:{color_right};'>R: {right_val}</span>, "
                            f"<span style='color:{color_left};'>L: {left_val}</span>, "
                            f"<span style='color:{color};'>Δ: {abs_diff}%</span>"
                            )
        categories.append(formatted_name)
    
    #ranges = range
    #print(categories)
    #print(range)
    
    values_dict = {'50th bad':median_bad, '50th good': median_good, 'max good': max_good, 'max bad': max_bad, 'min bad': min_bad}
    #print(values_dict)

    
    # Normalize values per axis and close the loop by repeating first point
    normed_data = {}
    for label, vals in values_dict.items():
        # vals: e.g. [15, 80, 40, 7, 150, 25]
        norm = []
        for v, (lo, hi, ch_name) in zip(vals, ranges):

            norm.append((v - lo) / (hi - lo))

        norm.append(norm[0])
        normed_data[label] = norm
    
    #
    
    # Save original categories before closing the loop for bar chart
    original_categories = categories.copy()
    
    # Prepare theta (angles) and extend categories to close the loop
    theta = categories + [categories[0]]

#--------------------------------------------------------------------    

    categories = categories + [categories[0]]
    traces = []

    # Add colored lines between left and right points on each axis
    # Use the formatted category names (without the duplicate closing element)
    formatted_categories = categories[:-1]  # Remove the duplicate closing element
    
    for idx in range(len(formatted_categories)):
        right_val_norm = normed_data_uid['Right'][idx]
        left_val_norm = normed_data_uid['Left'][idx]
        
        right_val = single_uid['Right'][idx]
        left_val = single_uid['Left'][idx]
        abs_diff = percent_diff(right_val, left_val)
        
        # Choose color based on difference
        line_color = "red" if abs_diff > 10 else "green"
        line_width = 4
        
        # Create a line segment between the two points on this axis
        traces.append({
            "type": "scatterpolar",
            "r": [right_val_norm, left_val_norm],
            "theta": [formatted_categories[idx], formatted_categories[idx]],
            "mode": "lines",
            "line": {
                "color": line_color,
                "width": line_width
            },
            "showlegend": False,
            "hoverinfo": "skip"
        })

    for label, norm_vals in normed_data_uid.items():
        # Marker colors based on normalized value
        marker_colors = norm_vals[:-1]  # exclude duplicate
        # Text annotations are actual values
        text_vals = [str(v) for v in single_uid[label]] + [str(single_uid[label][0])]
        trace = {
                "type": "scatterpolar",
                "r": norm_vals,
                "theta": theta,
                "name": f"Test subject #{uid_list.index(uid)} {label}, side",
                "text": text_vals,
                "textposition": "top center"
        }
        traces.append(trace)   


    max_good_norm = list(normed_data['max good'])
    max_good_norm += [max_good_norm[0]]

    max_bad_norm = list(normed_data['max bad'])
    max_bad_norm += [max_bad_norm[0]]
    
    
    median_good_norm = list(normed_data['50th good'])
    median_good_norm += [median_good_norm[0]]

    median_bad_norm = list(normed_data['50th bad'])
    median_bad_norm += [median_bad_norm[0]]
    
    
    min_bad_norm = list(normed_data['min bad'])
    min_bad_norm += [min_bad_norm[0]]
    if uid_inj != 'NO RECORDS':

        # Add concentric sections for red, yellow, and green ranges
        traces.append({
            "type": "scatterpolar",
            "r": [0.3] * len(theta),  # Radius for the red section
            "theta": theta,
            "fill": "toself",
            "fillcolor": "rgba(255, 0, 0, 0.3)",  # Red color
            "line": {"color": "rgba(0, 0, 0, 0)"},  # No border
            "name": "0.0 - 0.3 (Red)",
            "showlegend": False
        })

        traces.append({
            "type": "scatterpolar",
            "r": [0.6] * len(theta),  # Radius for the yellow section
            "theta": theta,
            "fill": "tonext",  # Fill between this and the previous trace
            "fillcolor": "rgba(255, 255, 0, 0.3)",  # Yellow color
            "line": {"color": "rgba(0, 0, 0, 0)"},  # No border
            "name": "0.3 - 0.6 (Yellow)",
            "showlegend": False
        })

        traces.append({
            "type": "scatterpolar",
            "r": [1.0] * len(theta),  # Radius for the green section
            "theta": theta,
            "fill": "tonext",  # Fill between this and the previous trace
            "fillcolor": "rgba(0, 255, 0, 0.3)",  # Green color
            "line": {"color": "rgba(0, 0, 0, 0)"},  # No border
            "name": "0.6 - 1.0 (Green)",
            "showlegend": False
        })


    layout = {
        "polar": {
            "radialaxis": {
                "visible": True,
                "range": [0, 1]
            }
        },
        "width": width,
        "height": height,
        "showlegend": True,
        "showticklabels": False,
        "title": f"Exe. type: {type_of_test}, subject: {time_subject}, cohort: {time_cohort}, Injured side: {uid_inj}"
    }
    
    # Create bar chart version with the same data
    bar_chart = create_bar_chart_from_polar(traces, original_categories, single_uid, normed_data_uid, uid, uid_list, time_subject, type_of_test, uid_inj, width, height, display_mode=display_mode, all_info=all_info)
    
    return {"traces": traces, "layout": layout, "bar_chart": bar_chart}

# plot function for all time periods
def func_plot_3(uid,
                time_subject, 
                time_cohort, 
                temp_ch_list,
                type_of_test = 'unilateral_squat', 
                ch_type = 'mobility',
                width=500, 
                height=500):
#-------- data preprocessing ----------------    

    global uid_list
    global time_points

    time = '053-Time after Surgery'
    gender = '005-Gender'
    injure = 'injured_side'
    global time_points 

    # Initialize variables
    single_uid = {'Right': [], 'Left': []}
    normed_data_uid = {}
    normed_data = {}
    uid_inj = 'NO RECORDS'

    #uid = uid_list[0]
    #temp_ch_list = unilateral_squat_test_all
    #time_v = '9 Month'
    all_info = {}
    for i in temp_ch_list:
    
        temp = i[0].split('__')
        if 'pct' in i[0]:
            axis = temp[2] + '__' + temp[3] + '_pct'
        else:
            axis = temp[2] + '__' + temp[3]
    
        # Define a dictionary with keys from time_points and empty lists as values
        time_cohort_dict = {}
        # Iterate through the time_points and filter the DataFrame for each time point
        for tp in time_points:
            y_temp_spider = df_test[[time, *i, injure]]
            y_temp = y_temp_spider[y_temp_spider[time] == tp]

            # Collect good and bad values
            good_values = y_temp[y_temp[injure] == 'Left'][i[1]].values.tolist() + y_temp[y_temp[injure] == 'Right'][i[0]].values.tolist()
            good_values_ps = pd.Series(good_values)

            bad_values = y_temp[y_temp[injure] == 'Left'][i[0]].values.tolist() + y_temp[y_temp[injure] == 'Right'][i[1]].values.tolist()
            bad_values_ps = pd.Series(bad_values)
            
            # Update the dictionary with calculated statistics for the current time point
            time_cohort_dict[tp] = {
                'keys': i,
                'bad': stat_calc(bad_values_ps, [0.02,0.98]),
                'good': stat_calc(good_values_ps, [0.02,0.98])
            }

        # Nest the keys: first level is axis, next level is time, then statistics
        all_info[axis] = time_cohort_dict

    # Calculate statistics for all keys in time_cohort_dict

    # Calculate range for all channels in all_info
    ranges = {}
    # Calculate range for all channels using all time periods
    for k in all_info:
        # For each channel, collect min and max from bad_stats only across all time periods
        min_vals = []
        max_vals = []
        for tp in time_points:
            try:
                stats = all_info[k][tp]
                bad_stats = stats['bad']
                good_stats = stats['good']
                min_vals.append(bad_stats['min'])
                max_vals.append(bad_stats['max'])
            except KeyError:
                continue

        lo = min(min_vals)
        hi = max(max_vals)
        ranges[k] = [lo, hi]

    print(f"[DEBUG] axis range values plot_3: {ranges}")
    #categories = [k for k in all_info]
#    range_abs = [min([r[0] for r in ranges]), max([r[1] for r in ranges])]

        #--------------------------------- single UID -------------------------
    
    
    single_uid = {'Right':[], 'Left':[] }
    for ch in temp_ch_list:
        y_temp_spider = df_test[[time, *ch, injure, 'uid']]
        y_temp = y_temp_spider[y_temp_spider[time] == time_subject]
        
        # Debugging output to log the state of variables
#        print("Processing channel:", ch)
#        print("Filtered DataFrame for time_v:", y_temp)
 #       print("UID being processed:", uid)

        # Safely access the first element of the list, or assign a default value if the list is empty
        right_values = list(y_temp[y_temp['uid'] == uid][ch[1]])
        if not right_values:
            right_values = [0]
        left_values = list(y_temp[y_temp['uid'] == uid][ch[0]])
        if not left_values:
            left_values = [0]

#        print("Right values:", right_values)
#        print("Left values:", left_values)

        single_uid['Right'].append(right_values[0] if right_values else None)
        single_uid['Left'].append(left_values[0] if left_values else None)
    
    print(f"[DEBUG] single_uid raw values plot_3: {single_uid}")

    uid_inj_list = list(y_temp[y_temp['uid'] == uid][injure])
    uid_inj = uid_inj_list[0] if uid_inj_list else 'NO RECORDS'
    
    # build an ordered list of axes and corresponding ranges
    ordered_axes = list(all_info.keys())
    ordered_ranges = [ranges[axis] for axis in ordered_axes]
    
    normed_data_uid = {}
#    print(label)

    def get_normalized_median_bad(all_info, ranges, time_points):
        """
        Calculate normalized median 'bad' values for all channels and all time periods in all_info.
        Returns a dict: {channel: {time_point: normalized_median_bad}}
        """
        normalized_medians = {}
        for axis in all_info:
            normalized_medians[axis] = {}
            for tp in time_points:
                try:
                    median_bad = all_info[axis][tp]['bad']['median']
                    lo, hi = ranges[axis]
                    # Avoid division by zero
                    if hi != lo:
                        norm_val = (median_bad - lo) / (hi - lo)
                    else:
                        norm_val = 0.0
                    normalized_medians[axis][tp] = norm_val
                except Exception:
                    normalized_medians[axis][tp] = None
        return normalized_medians
    
    all_bad_norms = get_normalized_median_bad(all_info, ranges, time_points)
    
    for label, vals in single_uid.items():
        # vals: e.g. [15, 80, 40, 7, 150, 25]
        norm = []
        for v, (lo, hi), ch_name in zip(vals, ordered_ranges, ordered_axes):
            # Normalize by the range of the uninjured side
            if 'alignment' in ch_name:
                norm.append(1 - ((v - lo) / (hi - lo)))
            else:
                norm.append((v - lo) / (hi - lo))
        normed_data_uid[label] = norm
    # close the loop for each UID trace so the radar line connects back
    for label in normed_data_uid:
        normed_data_uid[label].append(normed_data_uid[label][0])

    #categories = [k for k in all_info]
    categories = ordered_axes

    
     
    # Prepare theta (angles) and extend categories to close the loop
    theta = categories + [categories[0]]

#--------------------------------------------------------------------    

#    categories = categories + [categories[0]]
    traces = []

    if uid_inj == 'Right':
        norm_vals = normed_data_uid['Right']
        label = 'Right'
    else:
        norm_vals = normed_data_uid['Left']
        label = 'Left'


#        text_vals = [str(v) for v in single_uid[label]] + [str(single_uid[label][0])]
    trace = {
            "type": "scatterpolar",
            "r": norm_vals,
            "theta": theta,
            "name": f"Test subject #{uid_list.index(uid)} {label}, side",
#                "text": text_vals,
            "textposition": "top center"
    }
    traces.append(trace)   
    period_colors = {
        "3 Month":  "rgba(255, 0, 0, 0.8)",      # red
        "6 Month":  "rgba(255, 165, 0, 0.8)",    # orange
        "9 Month":  "rgba(255, 255, 0, 0.8)",    # yellow
        "12 Month": "rgba(0, 128, 0, 0.8)",      # green
    }

    # Set test subject trace color to black
    traces[0]["line"] = {"color": "black", "width": 3}
    # Build exactly 4 traces: one per time period, across all axes
    for tp in time_points:
        series = []
        for axis in ordered_axes:
            val = all_bad_norms.get(axis, {}).get(tp)
            # clamp to [0, 1] so axis min can safely be 0.0
            v = 0.0 if val is None else float(val)
            series.append(min(1.0, max(0.0, v)))
        # close the loop
        r_vals = series + [series[0]]
        traces.append({
            "type": "scatterpolar",
            "r": r_vals,
            "theta": theta,
            "mode": "lines",
            "name": f"Median bad ({tp})",
            "legendgroup": f"median-{tp}",
            "showlegend": True,
            "line": {"color": period_colors.get(tp, "rgba(0,0,0,0.6)"), "width": 2}
        })

    # --- Auto-zoom top; force radial axis minimum to 0.0 ---
    r_max = 0.0
    for t in traces:
        for rv in t.get("r", []):
            try:
                v = float(rv)
            except (TypeError, ValueError):
                continue
            if pd.notna(v):
                r_max = max(r_max, v)
    # fallback and padding
    if r_max <= 0.0:
        r_max = 1.0
    else:
        r_max = min(1.0, r_max + 0.05 * r_max)  # small headroom

    layout = {
        "polar": {
            "radialaxis": {
                "visible": True,
                "autorange": False,
                "range": [0.0, r_max]  # min fixed at 0.0
            }
        },
        "width": width,
        "height": height,
        "showlegend": True,
        "showticklabels": False,
        "title": f"Exe. type: {type_of_test}, subject: {time_subject}, Injured side: {uid_inj}"
    }
    
    # Create bar chart version with the same data - show progress across time periods
    bar_traces = []
    
    # Add test subject's injured side values first
    if uid_inj == 'Right':
        subject_values = normed_data_uid['Right'][:-1]  # Remove closing duplicate
        side_label = 'Right (Injured)'
    else:
        subject_values = normed_data_uid['Left'][:-1]  # Remove closing duplicate
        side_label = 'Left (Injured)'
    
    bar_traces.append({
        "type": "bar",
        "x": ordered_axes,
        "y": subject_values,
        "name": f"Subject #{uid_list.index(uid)} - {side_label}",
        "marker": {"color": "black"}
    })
    
    # Add cohort median bad values for each time period
    for tp in time_points:
        series = []
        for axis in ordered_axes:
            val = all_bad_norms.get(axis, {}).get(tp)
            v = 0.0 if val is None else float(val)
            series.append(min(1.0, max(0.0, v)))
        
        bar_traces.append({
            "type": "bar",
            "x": ordered_axes,
            "y": series,
            "name": f"Cohort Median ({tp})",
            "marker": {"color": period_colors.get(tp, "gray")}
        })
    
    bar_layout = {
        "barmode": "group",
        "width": width,
        "height": height,
        "title": f"Bar Chart: {type_of_test} - Subject vs Cohort Progress",
        "xaxis": {
            "title": "Metrics",
            "tickangle": -45
        },
        "yaxis": {
            "title": "Normalized Value",
            "range": [0, 1.1]
        },
        "margin": {
            "b": 150
        },
        "showlegend": True,
        "shapes": [
            # Red zone (0.0 - 0.3)
            {
                "type": "rect",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "y0": 0.0,
                "x1": 1,
                "y1": 0.3,
                "fillcolor": "rgba(255, 0, 0, 0.3)",
                "line": {"width": 0},
                "layer": "below"
            },
            # Yellow zone (0.3 - 0.6)
            {
                "type": "rect",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "y0": 0.3,
                "x1": 1,
                "y1": 0.6,
                "fillcolor": "rgba(255, 255, 0, 0.3)",
                "line": {"width": 0},
                "layer": "below"
            },
            # Green zone (0.6 - 1.0)
            {
                "type": "rect",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "y0": 0.6,
                "x1": 1,
                "y1": 1.0,
                "fillcolor": "rgba(0, 255, 0, 0.3)",
                "line": {"width": 0},
                "layer": "below"
            }
        ]
    }
    
    bar_chart = {"traces": bar_traces, "layout": bar_layout}
    
    return {"traces": traces, "layout": layout, "bar_chart": bar_chart}

@app.route("/")
@login_required
def home():
    if 'email' in session:
        return render_template("home.html")
    else:
        return '<a href="/login">Login with Google</a>'

# In-memory cache for chart data
chart_cache= {}

@app.route("/get_chart_data", methods=["GET"])
@login_required
def get_chart_data():
    global uid_list
    global time_points
    global unilateral_squat_test_all 

    time_cohort = request.args.get("dropdownTimeCohort", "9 Month")
    time_subject = request.args.get("optionTime", "9 Month")
    idx_str = request.args.get("uid", "0")
    try:
        idx = int(idx_str)
    except ValueError:
        idx = 0
    idx = max(0, min(idx, len(uid_list) - 1))
    uid = uid_list[idx]
    exe_type = request.args.get("EXE", "unilateral_squat")

    # Select test type
    if exe_type == 'unilateral_squat':
        temp_ch_list = unilateral_squat_test_all
    elif exe_type == 'bilateral_squat': 
        temp_ch_list = bilateral_squat_all
    elif exe_type == 'lateral_launch_all':
        temp_ch_list = lateral_launch_all
    elif exe_type == 'stork_stance_all':
        temp_ch_list = stork_stance_all
    elif exe_type == 'step_down_all':
        temp_ch_list = step_down_all
    elif exe_type == 'front_lunge_all':
        temp_ch_list = front_lunge_all

    display_mode = request.args.get("displayMode", "absolute")
    
    # Don't use cache for now to ensure fresh data with bar_chart
    # cache_key = (uid, time_subject, time_cohort, exe_type)
    # if cache_key in chart_cache:
    #     chart_data = chart_cache[cache_key]
    # else:
    #     chart_data = func_plot_3(uid, time_subject, time_cohort, temp_ch_list, exe_type, width=700, height=700)
    #     chart_cache[cache_key] = chart_data
    
    # Generate fresh data with bar charts
    chart_data = func_plot_3(uid, time_subject, time_cohort, temp_ch_list, exe_type, width=700, height=700)

    # For chart2, call func_plot_2 with display_mode parameter
    chart_data_2 = func_plot_2(uid, time_subject, time_cohort, temp_ch_list, exe_type, width=700, height=700, display_mode=display_mode)
    
    print(f"[DEBUG API] chart1 keys: {chart_data.keys()}")
    print(f"[DEBUG API] chart2 keys: {chart_data_2.keys()}")
    print(f"[DEBUG API] chart1 has bar_chart: {'bar_chart' in chart_data}")
    print(f"[DEBUG API] chart2 has bar_chart: {'bar_chart' in chart_data_2}")

    return jsonify({
        "chart1": chart_data,
        "chart2": chart_data_2,
        "options": time_points,
        "timePoints": time_points,
        "exe_type": ["unilateral_squat", "bilateral_squat", "lateral_launch_all", "stork_stance_all", "step_down_all", "front_lunge_all"],
        "UID_list": list(range(len(uid_list)))
    })

@app.route('/login')
def login():
    redirect_url = url_for('authorize', _external=True)
    print(f"Redirecting to: {redirect_url}")
    # force consent prompt and request offline access
    return google.authorize_redirect(redirect_url, prompt='consent', access_type='offline')

# Define which Google accounts are allowed to log in
ALLOWED_USERS = {
    'krivolapov.maksim@gmail.com',
    'mkrivolapov@darimotion.com',
    'snaggs100@gmail.com',
    'mchronert@darimotion.com',
    'dwassom@darimotion.com'
}

@app.route('/login/callback')
def authorize():
    print("Authorization callback triggered.")
    token = google.authorize_access_token()
    print(f"Access token: {token}")
    user_info = google.get('userinfo').json()
    print(f"User info: {user_info}")
    email = user_info.get('email')
    # Restrict login to allowed users
    if email not in ALLOWED_USERS:
        return "Access denied: unauthorized user", 403
    session['email'] = email
    return redirect('/')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/')

@app.route('/health')
def health_check():
    return 'OK', 200

@app.route('/ready')
def readiness_check():
    # Simple check: ensure data loaded into df_test
    try:
        if not df_test.empty:
            return 'READY', 200
    except NameError:
        pass
    return 'NOT READY', 503

# Error handlers
@app.errorhandler(401)
def unauthorized_error(e):
    return jsonify(error='Unauthorized access'), 401

@app.errorhandler(400)
def bad_request_error(e):
    return jsonify(error='Bad request'), 400

if __name__ == "__main__":
    app.run(debug=True)

#print("Looking for:", pkl_path)
#print("Exists:", os.path.exists(pkl_path))
