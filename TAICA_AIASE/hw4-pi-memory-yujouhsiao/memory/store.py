"""持久層：把記憶存成 JSON 檔，重啟後讀得回來，並用 id 去重。
load()/_persist()/add() 要你填。"""
from __future__ import annotations
import json
import os


class JsonStore:
    def __init__(self, path: str):
        self.path = path
        self.items: list[dict] = []
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            self.items = []
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.items = data if isinstance(data, list) else []
        except Exception:
            self.items = []

    def _persist(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)

    def add(self, obs: dict) -> bool:
        if any(o["id"] == obs["id"] for o in self.items):
            return False
        self.items.append(obs)
        self._persist()
        return True

    def all(self) -> list[dict]:
        return list(self.items)

    def clear(self) -> None:
        self.items = []
        self._persist()