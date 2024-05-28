import datetime



def current_time() -> str:
    return f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'
