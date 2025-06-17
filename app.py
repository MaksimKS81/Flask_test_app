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
def stat_calc(data_ps, filter = [0.05, 0.95]):
    # Calculate 5th and 95th percentiles for filtering
    lower_bound = data_ps.quantile(filter[0])
    upper_bound = data_ps.quantile(filter[1])
    
    # Filter data based on these percentiles
    filtered = data_ps[(data_ps >= lower_bound) & 
                     (data_ps <= upper_bound)]
    
    median_1 = round(filtered.quantile(0.50), 3)
    
    max_1 = filtered.max()
    
    min_1 = filtered.min()
    return {'max': max_1, 'median': median_1, 'min': min_1}

#-------------------------- Flask app setup ----------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
#------------------------------------------------------------------------

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
                time_v, 
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
        y_temp = y_temp_spider[y_temp_spider[time] == time_v]
        
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
        y_temp = y_temp_spider[y_temp_spider[time] == time_v]
        
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
    
    uid_inj_list = list(y_temp[y_temp['uid'] == uid][injure])
    uid_inj = uid_inj_list[0] if uid_inj_list else 'NO RECORDS'
    
 #   print('Injured:' + uid_inj)
    
    normed_data_uid = {}
    
    for label, vals in single_uid.items():
        # vals: e.g. [15, 80, 40, 7, 150, 25]
        norm = []
        for v, (lo, hi, ch_name) in zip(vals, ranges):
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
        "title": f"{ch_type} results, {type_of_test}, {time_v}, Injured side: {uid_inj} "
    }
    return {"traces": traces, "layout": layout}

@app.route("/")
@login_required
def home():
    if 'email' in session:
        return render_template("home.html")
    else:
        return '<a href="/login">Login with Google</a>'

# In-memory cache for chart data
chart_cache = {}

@app.route("/get_chart_data", methods=["GET"])
@login_required
def get_chart_data():
    # Suppose you receive two options from the frontend
    global uid_list
    global time_points
    global unilateral_squat_test_all 
 
    option1 = request.args.get("option1", "Option 1")
    time_v = request.args.get("optionTime", "9 Month")
    # uid = request.args.get("uid", uid_list[0])  
    # interpret uid param as index into uid_list
    idx_str = request.args.get("uid", "0")
    try:
        idx = int(idx_str)
    except ValueError:
        idx = 0
    idx = max(0, min(idx, len(uid_list) - 1))
    uid = uid_list[idx]
    exe_type = request.args.get("EXE", "unilateral_squat")


    print("UID:", uid)
    print("Time Point:", time_v)


    categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
    all_options = {
        "Option 1": [[20, 30, 40, 50, 60], [15, 25, 35, 45, 55]],
        "Option 2": [[10, 20, 30, 40, 50], [5, 15, 25, 35, 45]]
    }

    #selected_values1 = all_options.get(option1, all_options["Option 1"])
    #selected_values2 = all_options.get(option2, all_options["Option 1"])

    # Call func_plot with the correct arguments

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


    cache_key = (uid, time_v, exe_type)
    if cache_key in chart_cache:
        chart_data = chart_cache[cache_key]
    else:
        chart_data = func_plot(uid, time_v, temp_ch_list, exe_type, width=800, height=800)
        chart_cache[cache_key] = chart_data

    # Return cached results for both charts
    return jsonify({
        "chart1": chart_data,
        "chart2": chart_data,
        "options": list(all_options.keys()),
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
    'Snaggs100@gmail.com',
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
    app.run(debug=False)

#print("Looking for:", pkl_path)
#print("Exists:", os.path.exists(pkl_path))
