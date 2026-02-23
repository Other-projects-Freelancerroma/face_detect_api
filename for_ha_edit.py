import datetime
import os
import requests
from requests.auth import HTTPDigestAuth
import logging


log = logging.getLogger(__name__)


@pyscript_executor
def download_dahua_jpg(
    save_dir="/config/www/snapshots/",
    filename_prefix="dahua",
    url="http://172.20.22.9/cgi-bin/snapshot.cgi?channel=1&subtype=0",
    username="admin",
    password="PT8_dTdieYHTgAh"
):
    """
    Скачивает скриншот и сохраняет как .jpg. Возвращает путь к файлу
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.jpg"
    save_path = os.path.join(save_dir, filename)

    auth = HTTPDigestAuth(username, password)
    response = requests.get(url, auth=auth, timeout=12, stream=True)
    response.raise_for_status()

    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    size = os.path.getsize(save_path)
    log.info(f"Снимок сохранён → {save_path} ({size:,} байт)")

    return {
        "ok": True,
        "jpg_path": save_path,
        "size_bytes": size
    }


@pyscript_executor
def send_to_deepface_server(jpg_path):
    """
    Отправляет файл на DeepFace-сервер и возвращает результат распознавания
    """

    url = "http://172.20.22.12:5000/recognize"          # ← измени на актуальный адрес и порт
    # url = "http://192.168.1.55:5000/recognize"        # пример
    # url = "https://your-domain.com/recognize"         # если через nginx + ssl

    files = {
        "image": (
            os.path.basename(jpg_path),
            open(jpg_path, "rb"),
            "image/jpeg"
        )
    }
    headers = {}  # можно добавить: headers = {"X-API-Key": "твой_ключ"}

    log.info(f"Отправка файла на распознавание → {jpg_path}")
    r = requests.post(url, files=files, headers=headers, timeout=45)

    log.info(f"Ответ получен. Статус: {r.status_code}")

    if not r.ok:
        return {
            "ok": False,
            "status_code": r.status_code,
            "error": f"HTTP {r.status_code}: {r.text[:200]}"
        }

    data = r.json()

    return {
        "ok": True,
        "status_code": r.status_code,
        "result": data
    }


@service
def dahua_snapshot_to_deepface(
    delete_after: bool = False,
    jpg_dir="/config/www/snapshots/",
    jpg_prefix="dahua_recog"
):
    """
    Скачивание снимка с камеры, отправка и возвращение результата
    """

    # Скачивание снимка
    jpg_result = download_dahua_jpg(
        save_dir=jpg_dir,
        filename_prefix=jpg_prefix
    )

    jpg_path = jpg_result["jpg_path"]
    log.info(f"Работаем с файлом: {jpg_path}")

    # Отправка на сервер DeepFace
    api_result = send_to_deepface_server(jpg_path)

    # Подготовка результата
    final_result = {
        "ok": api_result["ok"],
        "jpg_path": jpg_path,
        "api_status": api_result.get("status_code"),
        "error": api_result.get("error"),
    }

    if api_result["ok"]:
        data = api_result["result"]
        final_result["recognition"] = data

        status = data.get("status", "unknown")

        if status == "recognized":
            final_result["person"] = data.get("person")
            final_result["distance"] = data.get("distance")
            log.info(
                f"Распознано: {data.get('person')} (dt {data.get('distance')})"
            )
        elif status == "not_recognized":
            final_result["best_match"] = data.get("best_match")
            final_result["best_distance"] = data.get("best_distance")
            log.info(
                "Лицо найдено, но не совпало ни с одной фотографией"
            )
        elif status == "no_face":
            log.info("Лицо на снимке не обнаружено")
        elif status == "error":
            log.warning(f"Ошибка сервера: {data.get('message')}")
        else:
            log.warning(f"Неизвестный статус от сервера: {status}")

    # Очистка файла после обработки (по желанию)
    if delete_after and os.path.exists(jpg_path):
        try:
            os.remove(jpg_path)
            log.info(f"Временный файл удалён: {jpg_path}")
        except Exception as e:
            log.warning(f"Не удалось удалить файл {jpg_path}: {e}")

    return final_result
