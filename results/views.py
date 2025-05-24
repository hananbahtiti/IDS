import logging
from django.shortcuts import render
from django.conf import settings
from .models import IntrusionResult
from django.utils.dateparse import parse_datetime
from django.utils.timezone import make_aware, is_naive
import requests
from django.http import JsonResponse
from .models import IntrusionResult
from django.forms.models import model_to_dict
from django.views.decorators.csrf import csrf_exempt
import os
import csv
import io
from .models import IntrusionResult

logger = logging.getLogger(__name__)


def get_intrusion_data_from_db(request):
    results = IntrusionResult.objects.order_by('-timestamp')[:100]
    rows = [model_to_dict(result) for result in results]
    return JsonResponse({'rows': rows})

def fetch_intrusion_data_api(request):
    fastapi_url = 'https://14eb-35-243-225-44.ngrok-free.app/predict_all'

    try:
        response = requests.get(fastapi_url)

        if response.status_code == 200:
            try:
                data = response.json()
                results = data.get('results', [])

                rows = []
                for result in results:
                    for item in result:
                        row_index = item.get('row_index', -1)
                        attack_cat = item.get('attack_cat', 'N/A')
                        mse = item.get('mse', 0.0)
                        result_val = item.get('result', 'N/A')
                        ct_src_dport_ltm = item.get('ct_src_dport_ltm', 'N/A')
                        rate = item.get('rate', 'N/A')
                        dwin = item.get('dwin', 'N/A')
                        dload = item.get('dload', 'N/A')
                        swin = item.get('swin', 'N/A')
                        ct_dst_sport_ltm = item.get('ct_dst_sport_ltm', 'N/A')
                        ct_state_ttl = item.get('ct_state_ttl', 'N/A')
                        sttl = item.get('sttl', 'N/A')
                        timestamp_raw = item.get('timestamp', None)
                        src = item.get('src', None)
                        proto = item.get('proto', None)
                        state = item.get('state', None)

                        # التعامل مع timestamp
                        ts = parse_datetime(timestamp_raw) if timestamp_raw else None
                        if ts and is_naive(ts):
                            ts = make_aware(ts)

                        # إنشاء السجل

                        # تحقق من وجود السجل مسبقًا
                        exists = IntrusionResult.objects.filter(
                            row_index=row_index,
                            attack_cat=attack_cat,
                            mse=mse,
                            result=result_val,
                            ct_src_dport_ltm=ct_src_dport_ltm,
                            rate=rate,
                            dwin=dwin,
                            dload=dload,
                            swin=swin,
                            ct_dst_sport_ltm=ct_dst_sport_ltm,
                            ct_state_ttl=ct_state_ttl,
                            sttl=sttl,
                            timestamp=ts,
                            src=src,
                            proto=proto,
                            state=state,
                        ).exists()

                        # إذا لم يكن موجودًا أضفه
                        if not exists:
                            IntrusionResult.objects.create(
                                row_index=row_index,
                                attack_cat=attack_cat,
                                mse=mse,
                                result=result_val,
                                ct_src_dport_ltm=ct_src_dport_ltm,
                                rate=rate,
                                dwin=dwin,
                                dload=dload,
                                swin=swin,
                                ct_dst_sport_ltm=ct_dst_sport_ltm,
                                ct_state_ttl=ct_state_ttl,
                                sttl=sttl,
                                timestamp=ts,
                                src=src,
                                proto=proto,
                                state=state,
                            )

                        
                        

                        rows.append({
                            'row_index': row_index,
                            'attack_cat': attack_cat,
                            'mse': mse,
                            'result': result_val,
                            'ct_src_dport_ltm': ct_src_dport_ltm,
                            'rate': rate,
                            'dwin': dwin,
                            'dload': dload,
                            'swin': swin,
                            'ct_dst_sport_ltm': ct_dst_sport_ltm,
                            'ct_state_ttl': ct_state_ttl,
                            'sttl': sttl,
                            'timestamp': timestamp_raw,
                            'src':src,
                            'proto':proto,
                            'state':state,
                        })

                return JsonResponse({'rows': rows})

            except ValueError as e:
                logger.error(f"JSON decode error: {e}")
                return JsonResponse({'error': 'Invalid JSON from FastAPI'}, status=500)
        else:
            return JsonResponse({'error': 'Failed to fetch data from FastAPI'}, status=502)

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return JsonResponse({'error': str(e)}, status=503)




# HTML View: Renders data in template
def fetch_intrusion_data_page(request):
    rows = IntrusionResult.objects.order_by('-timestamp')[:100]
    return render(request, 'predict_result.html', {'rows': rows})












"""
import os
import csv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
from datetime import datetime

@csrf_exempt
def save_csv(request):
    if request.method == 'POST':
        body = json.loads(request.body)
        data = body.get("data", [])

        # تحديد المجلد
        export_dir = os.path.join(settings.MEDIA_ROOT, "exports")
        os.makedirs(export_dir, exist_ok=True)

        # اسم الملف
        filename = f"intrusion_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(export_dir, filename)

        # كتابة البيانات في CSV
        with open(file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        return JsonResponse({"status": "success", "file_path": f"/media/exports/{filename}"})
    
    return JsonResponse({"status": "failed", "reason": "Invalid method"})"""



import os
import csv
import json
import requests
from datetime import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

@csrf_exempt
def save_csv(request):
    if request.method == 'POST':
        body = json.loads(request.body)
        data = body.get("data", [])

        print(f'print data: {data}')

        if not data:
            return JsonResponse({"status": "failed", "reason": "No data provided"})

        export_dir = os.path.join(settings.MEDIA_ROOT, "exports")
        os.makedirs(export_dir, exist_ok=True)

        filename = f"intrusion_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(export_dir, filename)
        print(f'print file_path: {file_path}')
        # إنشاء CSV
        with open(file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        # إرسال الملف إلى FastAPI
        fastapi_url = "https://200d-34-173-173-130.ngrok-free.app:8000/train"  # عدّل حسب عنوان FastAPI

        print(f'print file: {f}')

        try:
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f, 'text/csv')}
                response = requests.post(fastapi_url, files=files)
                print(f'print response: {response}, {files}')

            if response.status_code == 200:
                return JsonResponse({
                    "status": "success",
                    "message": "CSV saved and sent to FastAPI",
                    "fastapi_response": response.json()
                })
            else:
                return JsonResponse({
                    "status": "partial_success",
                    "message": "CSV saved but failed to send to FastAPI",
                    "fastapi_response": response.text
                })

        except Exception as e:
            return JsonResponse({
                "status": "partial_success",
                "message": "CSV saved but error sending to FastAPI",
                "error": str(e)
            })

    return JsonResponse({"status": "failed", "reason": "Invalid method"})