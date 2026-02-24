import datetime
import os
import requests
from requests.auth import HTTPDigestAuth
import logging
import time


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
    log.warning(
        f"Снимок сохранён → {save_path} ({size:,} байт)"
    )
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
    url = "http://172.20.22.12:5000/recognize"
    files = {
        "image": (
            os.path.basename(jpg_path),
            open(jpg_path, "rb"),
            "image/jpeg"
        )
    }
    headers = {}
    log.warning(
        f"Отправка файла на распознавание → {jpg_path}"
    )
    r = requests.post(url, files=files, headers=headers, timeout=45)
    log.warning(
        f"Ответ получен. Статус: {r.status_code}"
    )
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
    с повторными попытками при "no_face" (до 5 раз, с паузой 30 сек)
    """
    max_attempts = 5
    final_result = None
    face_detected = False
    last_jpg_path = None

    for attempt in range(1, max_attempts + 1):
        jpg_result = download_dahua_jpg(
            save_dir=jpg_dir,
            filename_prefix=jpg_prefix
        )
        jpg_path = jpg_result["jpg_path"]
        last_jpg_path = jpg_path
        log.warning(
            f"Работаем с файлом: {jpg_path} (попытка {attempt}/{max_attempts})"
        )
        api_result = send_to_deepface_server(jpg_path)
        current_result = {
            "ok": api_result["ok"],
            "jpg_path": jpg_path,
            "api_status": api_result.get("status_code"),
            "error": api_result.get("error"),
            "attempt": attempt
        }

        if api_result["ok"]:
            data = api_result["result"]
            current_result["recognition"] = data
            status = data.get("status", "unknown")

            if status == "recognized":
                current_result["person"] = data.get("person")
                current_result["distance"] = data.get("distance")
                log.warning(
                    f"Распознано: {data.get('person')}"
                    f"(distance {data.get('distance')})"
                )
                face_detected = True

            elif status == "not_recognized":
                current_result["best_match"] = data.get("best_match")
                current_result["best_distance"] = data.get("best_distance")
                log.warning(
                    "Лицо найдено, но не совпало ни с одной фотографией"
                )
                face_detected = True

            elif status == "no_face":
                log.warning(
                    "Лицо на снимке не обнаружено"
                )

            elif status == "error":
                log.warning(
                    f"Ошибка сервера: {data.get('message')}"
                )

            else:
                log.warning(
                    f"Неизвестный статус от сервера: {status}"
                )

        if face_detected:
            final_result = current_result
            break

        if status == "no_face" and attempt < max_attempts:
            log.warning(
                f"Лицо не обнаружено."
                f"Повтор через 30 секунд (попытка {attempt + 1})"
            )
            time.sleep(30)

        elif not api_result["ok"] or status in ("error", "unknown"):
            final_result = current_result
            break

        final_result = current_result

    # Очистка файла после обработки (по желанию, только последнего)
    if delete_after and last_jpg_path and os.path.exists(last_jpg_path):
        try:
            os.remove(last_jpg_path)
            log.warning(
                f"Временный файл удалён: {last_jpg_path}"
            )
        except Exception as e:
            log.warning(
                f"Не удалось удалить файл {last_jpg_path}: {e}"
            )

    # Уведомление в интерфейсе HA
    notify_title = "Dahua - Распознавание лица"
    notify_message = (
        f"Обработка завершена после {attempt} попыток\n"
        f"Файл: {os.path.basename(last_jpg_path)}\n\n"
    )

    if not final_result["ok"]:
        notify_message += (
            f"**Ошибка сервера DeepFace:**\n"
            f"{final_result.get('error', 'неизвестная ошибка')}"
        )

    else:
        data = final_result.get("recognition", {})
        status = data.get("status", "unknown")

        if status == "recognized":
            person = data.get("person", "неизвестно")
            dist = data.get("distance", "?")
            notify_message += (
                f"**Распознано:** {person}\nРасстояние: {dist}"
            )

        elif status == "not_recognized":
            notify_message += (
                "Лицо найдено, но **не совпало** ни с одной записью в базе"
            )

        elif status == "no_face":
            notify_message += (
                "Лицо **не обнаружено** после 5 попыток"
            )

        else:
            notify_message += (
                f"Неизвестный статус от сервера: {status}"
            )

    notify_message += (
        f"\n\nВремя: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    if not delete_after and os.path.exists(last_jpg_path):
        image_url = last_jpg_path.replace("/config/www/", "/local/")
        notify_message += (
            f"\n\n![Снимок с камеры]({image_url})"
        )

    service.call(
        "persistent_notification",
        "create",
        notification_id="dahua_face_recognition_last",
        title=notify_title,
        message=notify_message,
    )

    log.warning(
        "Persistent notification отправлено: dahua_face_recognition_last"
    )

    # Отправка в Telegram
    telegram_message = notify_message
    image_url = None
    if not delete_after and last_jpg_path and os.path.exists(last_jpg_path):
        relative_url = last_jpg_path.replace("/config/www/", "/local/")
        base_url = ""
        try:
            if hasattr(hass.config.api, 'base_url') and hass.config.api.base_url:
                base_url = hass.config.api.base_url.rstrip('/')
        except:
            pass

        if not base_url:
            base_url = "http://172.20.22.10:8123"

        image_url = f"{base_url}{relative_url}"

    if image_url:
        try:
            service.call(
                "telegram_bot",
                "send_photo",
                url=image_url,
                caption=telegram_message,
                parse_mode="Markdown"
            )
            log.warning(
                f"Фото отправлено в Telegram по URL: {image_url}"
            )
        except Exception as e:
            log.error(
                f"Ошибка отправки фото по URL в Telegram: {str(e)}"
            )
            service.call(
                "telegram_bot",
                "send_message",
                message=telegram_message + "\n\n(не удалось прикрепить фото)",
                parse_mode="Markdown",
            )
    else:
        try:
            service.call(
                "telegram_bot",
                "send_message",
                message=telegram_message,
                parse_mode="Markdown",
            )
            log.warning(
                "Отправлено только текстовое сообщение в Telegram"
            )
        except Exception as e:
            log.error(
                f"Ошибка отправки текста в Telegram: {str(e)}"
            )

    return final_result
