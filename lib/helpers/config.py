from decouple import Config, RepositoryEnv
import sys
import os 

def get_config():
    if getattr(sys, 'default', False):
        dot_env_file = os.path.join(os.path.dirname(__file__), 'default.env')
    else:
        dot_env_file = os.path.join(os.path.dirname(__file__), '../../.env')
    return Config(RepositoryEnv(dot_env_file))

config = get_config()