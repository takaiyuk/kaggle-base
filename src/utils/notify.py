from typing import Dict

import requests


class LINENotify:
    def __init__(self, env_path: str = ".env") -> None:
        try:
            self.token: str = self._read_env(env_path)["LINE_NOTIFY_API"]
        except FileNotFoundError as e:
            print(f"[WARNING]: Skip to send message beceause: {e}")
            self.token: str = ""

    def send_message(self, message: str, verbose: bool = True) -> None:
        line_notify_api = "https://notify-api.line.me/api/notify"
        payload = {"message": message}
        headers = {"Authorization": f"Bearer {self.token}"}
        requests.post(line_notify_api, data=payload, headers=headers)
        if verbose and self.token:
            print("message sent")

    def _read_env(self, path: str) -> Dict[str, str]:
        with open(path) as f:
            lines = f.readlines()
        env_dict = {
            line.split("=")[0]: line.split("=")[1].rstrip("\n") for line in lines
        }
        return env_dict
