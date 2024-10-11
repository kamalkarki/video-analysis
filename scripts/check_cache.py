# write a function to check value of each ENV variable set in .env file

# write a function to read config.json
import json

def read_config():
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        print("Error: config.json file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in config.json file.")
        return None




def check_env_variables():
    from dotenv import load_dotenv
    import os
    config = read_config()
    print(config)
    # Load environment variables from .env file
    load_dotenv()

    # Get all environment variables
    env_vars = os.environ

    # Print all environment variables
    for key, value in env_vars.items():
        if key in config:
            print(f"{key}: {value}")


check_env_variables()
