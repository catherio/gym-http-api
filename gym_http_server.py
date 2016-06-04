from flask import Flask, request, jsonify
from functools import wraps
import uuid
import gym

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########## Container for environments ##########
class Envs(object):
    """
    Container and manager for the environments instantiated
    on this server.

    When a new environment is created, such as with
    envs.create('CartPole-v0'), it is stored under a short
    identifier (such as '3c657dbc'). Future API calls make
    use of this instance_id to identify which environment
    should be manipulated.
    """
    def __init__(self):
        self.envs = {}
        self.id_len = 8

    def _lookup_env(self, instance_id):
        try:
            return self.envs[instance_id]
        except KeyError:
            raise InvalidUsage('Instance_id {} unknown'.format(instance_id))

    def create(self, env_id):
        try:
            env = gym.make(env_id)
        except gym.error.Error:
            raise InvalidUsage('Attempted to look up malformed environment ID')

        instance_id = str(uuid.uuid4().hex)[:self.id_len]
        self.envs[instance_id] = env
        return instance_id

    def check_exists(self, instance_id):
        return instance_id in self.envs
    
    def list_all(self):
        return dict([(instance_id, env.spec.id) for (instance_id, env) in self.envs.items()])

    def reset(self, instance_id):
        env = self._lookup_env(instance_id)
        obs = env.reset()
        return env.observation_space.to_jsonable(obs)

    def step(self, instance_id, action):
        env = self._lookup_env(instance_id)
        action_from_json = int(env.action_space.from_jsonable(action))
        [observation, reward, done, info] = env.step(action_from_json)
        obs_jsonable = env.observation_space.to_jsonable(observation)
        return [obs_jsonable, reward, done, info]

    def get_action_space_info(self, instance_id):
        env = self._lookup_env(instance_id)
        return self._get_space_properties(env.action_space)

    def get_observation_space_info(self, instance_id):
        env = self._lookup_env(instance_id)
        return self._get_space_properties(env.observation_space)

    def _get_space_properties(self, space):
        info = {}
        info['name'] = space.__class__.__name__
        if info['name'] == 'Discrete':
            info['n'] = space.n
        elif info['name'] == 'Box':
            info['shape'] = space.shape
            info['low'] = space.to_jsonable(space.low)
            info['high'] = space.to_jsonable(space.high)
        return info
    
    def monitor_start(self, instance_id, directory, force, resume):
        env = self._lookup_env(instance_id)
        env.monitor.start(directory, force=force, resume=resume)

    def monitor_close(self, instance_id):
        env = self._lookup_env(instance_id)
        env.monitor.close()

########## App setup ##########
app = Flask(__name__)
envs = Envs()

########## Error handling ##########
class InvalidUsage(Exception):
    status_code = 400
    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

def catch_invalid_request_param(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError, e:
            logger.info('Caught invalid request param')
            raise InvalidUsage('A required request parameter was not provided')
    return wrapped

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

########## API route definitions ##########
@app.route('/v1/envs/', methods=['POST'])
@catch_invalid_request_param
def env_create():
    """
    Create an instance of the specified environment
    
    Parameters:
        - env_id: gym environment ID string, such as 'CartPole-v0'
    Returns:
        - instance_id: a short identifier (such as '3c657dbc')
        for the created environment instance. The instance_id is
        used in future API calls to identify the environment to be
        manipulated
    """
    env_id = request.get_json()['env_id']
    instance_id = envs.create(env_id)
    return jsonify(instance_id = instance_id)

@app.route('/v1/envs/', methods=['GET'])
def env_list_all():
    """
    List all environments running on the server

    Returns:
        - envs: dict mapping instance_id to env_id
        (e.g. {'3c657dbc': 'CartPole-v0'}) for every env
        on the server
    """
    all_envs = envs.list_all()
    return jsonify(all_envs = all_envs)

@app.route('/v1/envs/<instance_id>/check_exists/', methods=['POST'])
def env_check_exists(instance_id):
    """
    Determine whether the specified instance_id corresponds to
    a valid environment instance that has been created.
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - exists: True or False, indicating whether the given
        instance exists
    """
    exists = envs.check_exists(instance_id)
    return jsonify(exists = exists)

@app.route('/v1/envs/<instance_id>/reset/', methods=['POST'])
def env_reset(instance_id):
    """
    Reset the state of the environment and return an initial
    observation.
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - observation: the initial observation of the space
    """  
    observation = envs.reset(instance_id)
    return jsonify(observation = observation)

@app.route('/v1/envs/<instance_id>/step/', methods=['POST'])
@catch_invalid_request_param
def env_step(instance_id):
    """
    Run one timestep of the environment's dynamics.
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
        - action: an action to take in the environment
    Returns:
        - observation: agent's observation of the current
        environment
        - reward: amount of reward returned after previous action
        - done: whether the episode has ended
        - info: a dict containing auxiliary diagnostic information
    """  
    action = request.get_json()['action']
    [obs_jsonable, reward, done, info] = envs.step(instance_id, action)
    return jsonify(observation = obs_jsonable,
                    reward = reward, done = done, info = info)

@app.route('/v1/envs/<instance_id>/action_space/', methods=['GET'])
def env_action_space_info(instance_id):
    """
    Get information (name and dimensions/bounds) of the env's
    action_space
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
    - info: a dict containing 'name' (such as 'Discrete'), and
    additional dimensional info (such as 'n') which varies from
    space to space
    """  
    info = envs.get_action_space_info(instance_id)
    return jsonify(info = info)

@app.route('/v1/envs/<instance_id>/observation_space/', methods=['GET'])
def env_observation_space_info(instance_id):
    """
    Get information (name and dimensions/bounds) of the env's
    observation_space
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - info: a dict containing 'name' (such as 'Discrete'),
        and additional dimensional info (such as 'n') which
        varies from space to space
    """  
    info = envs.get_observation_space_info(instance_id)
    return jsonify(info = info)

@app.route('/v1/envs/<instance_id>/monitor/start/', methods=['POST'])
@catch_invalid_request_param
def env_monitor_start(instance_id):
    """
    Start monitoring.
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
        - force (default=False): Clear out existing training
        data from this directory (by deleting every file
        prefixed with "openaigym.")
        - resume (default=False): Retain the training data
        already in this directory, which will be merged with
        our new data
    
    (NOTE: the 'video_callable' parameter from the native
    env.monitor.start function is NOT implemented)
    """  
    request_data = request.get_json()

    directory = request_data['directory']
    force = request_data.get('force', False)
    resume = request_data.get('resume', False)

    envs.monitor_start(instance_id, directory, force, resume)
    return ('', 204)

@app.route('/v1/envs/<instance_id>/monitor/close/', methods=['POST'])
def env_monitor_close(instance_id):
    """
    Flush all monitor data to disk.
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
          for the environment instance
    """
    envs.monitor_close(instance_id)
    return ('', 204)

@app.route('/v1/upload/', methods=['POST'])
@catch_invalid_request_param
def upload():
    """
    Upload the results of training (as automatically recorded by
    your env's monitor) to OpenAI Gym.
    
    Parameters:
        - training_dir: A directory containing the results of a
        training run.
        - api_key: Your OpenAI API key
        - algorithm_id (default=None): An arbitrary string
        indicating the paricular version of the algorithm
        (including choices of parameters) you are running.
        - writeup (default=None): A Gist URL (of the form
        https://gist.github.com/<user>/<id>) containing your
        writeup for this evaluation.
        """  
    request_data = request.get_json()

    training_dir = request_data['training_dir']
    api_key = request_data['api_key']
    algorithm_id = request_data.get('algorithm_id', None)
    writeup = request_data.get('writeup', None)
    ignore_open_monitors = request_data.get('ignore_open_monitors', False)

    try:
        gym.upload(training_dir, algorithm_id, writeup, api_key,
                   ignore_open_monitors)
        return ('', 204)
    except gym.error.AuthenticationError:
        raise InvalidUsage('You must provide an OpenAI Gym API key')

@app.route('/v1/shutdown/', methods=['POST'])
def shutdown():
    f = request.environ.get('werkzeug.server.shutdown')
    f()
    return 'Server shutting down'

if __name__ == '__main__':
    app.run()
