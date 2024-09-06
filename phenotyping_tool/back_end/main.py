import traceback
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime
from typing import List, Dict
import json
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import re
from pathlib import Path
import logging
import openai
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
from pydantic import BaseModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 允许的源
    allow_credentials=True,
    allow_methods=["*"],  # 允许的方法
    allow_headers=["*"],  # 允许的头
)
PLANT_DATA_DIR = "/Volumes/Elements SE/upload"

@app.get("/plants")
async def get_plants():
    try:
        with open("data/data_available.json", "r") as f:
            data = json.load(f)

        plant_structure = defaultdict(lambda: defaultdict(set))

        for item in data:
            plant_structure[item["crop_type"]][item["geno_type"]].add(item["plant_id"])

        result = {
            crop_type: {
                geno_type: list(plant_ids)
                for geno_type, plant_ids in geno_types.items()
            }
            for crop_type, geno_types in plant_structure.items()
        }

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/initialize-plant")
async def initialize_plant(plant_data: Dict):
    crop_type = plant_data["cropType"]
    geno_type = plant_data["genoType"]
    plant_id = plant_data["plantId"]

    plant_dir = os.path.join(PLANT_DATA_DIR, plant_id)
    print(f"Initializing plant. Plant directory: {plant_dir}")  # 添加日志

    if not os.path.exists(plant_dir):
        raise HTTPException(status_code=404, detail="Plant not found")

    dates = []
    sv_000_dir = os.path.join(plant_dir, "sv-000")
    if os.path.exists(sv_000_dir):
        for file in os.listdir(sv_000_dir):
            if file.endswith(".png"):
                parts = file.split("-")
                if len(parts) >= 5:
                    year = parts[2]
                    month = parts[3]
                    day = parts[4].split("_")[0]
                    date_str = f"{year}-{month}-{day}"
                    try:
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        formatted_date = date_obj.strftime("%Y-%m-%d")
                        if formatted_date not in dates:
                            dates.append(formatted_date)
                    except ValueError:
                        print(f"Skipping invalid date in filename: {file}")
    else:
        print(f"sv-000 directory not found: {sv_000_dir}")  # 添加日志

    dates.sort()

    return {
        "cropType": crop_type,
        "genoType": geno_type,
        "plantId": plant_id,
        "dates": dates
    }


class ReportRequest(BaseModel):
    plant_id: str
    date: str


@app.post("/generate-report")
async def generate_report(report_request: ReportRequest, background_tasks: BackgroundTasks):
    try:
        plant_id = report_request.plant_id
        date = report_request.date

        # 获取所有需要的数据
        images = await get_plant_images(plant_id, date)
        bud_info = await get_bud_info(plant_id, date)
        branch_info = await get_branch_info(plant_id, date)

        # 为每个视图获取分支分析数据
        branch_analysis = {}
        for view in ["sv-000", "sv-045", "sv-090"]:
            branch_analysis[view] = await get_branch_analysis(plant_id, date, view)

        # 使用 GPT 生成报告文本
        report_text = generate_report_text(plant_id, date, images, bud_info, branch_info, branch_analysis)

        # 创建 PDF
        pdf_path = create_pdf_report(plant_id, date, images, report_text, branch_analysis)

        # 在后台删除生成的 PDF 文件
        background_tasks.add_task(os.remove, pdf_path)

        return FileResponse(pdf_path, filename=f"plant_report_{plant_id}_{date}.pdf")

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


def generate_report_text(plant_id, date, images, bud_info, branch_info, branch_analysis):
    openai.api_key = 'sk-proj-CZG9alrAIn8hX0T1ZdNYT3BlbkFJ81775NW45QAFKj3DOfpU'

    # 汇总 bud_info
    total_buds = sum(bud_info.get('budCounts', {}).values())

    # 汇总 branch_info
    branch_summary = f"Total branches: {branch_info.get('totalBranches', 0)}, Main stem length: {branch_info.get('mainStemLength', 0):.2f}"

    # 汇总 branch_analysis
    branch_analysis_summary = {}
    for view, data in branch_analysis.items():
        if 'branches' in data:
            branch_analysis_summary[view] = {
                'branch_count': len(data['branches']),
                'avg_length': sum(b['length'] for b in data['branches']) / len(data['branches']) if data[
                    'branches'] else 0
            }

    prompt = f"""
    Generate a concise report for plant {plant_id} on {date}. Include:

    1. Plant ID: {plant_id}
    2. Date: {date}
    3. Total buds: {total_buds}
    4. Branch summary: {branch_summary}
    5. Branch analysis summary:
    {json.dumps(branch_analysis_summary, indent=2)}

    Provide a brief analysis of the plant's growth and development based on this data.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a plant analysis expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        n=1,
        temperature=0.7,
    )

    return response.choices[0].message['content']


def create_pdf_report(plant_id, date, images, report_text, branch_analysis):
    pdf_path = f"plant_report_{plant_id}_{date}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)

    styles = getSampleStyleSheet()
    styles['Heading1'].fontSize = 16
    styles['Heading1'].spaceAfter = 12
    styles['Heading2'].fontSize = 14
    styles['Heading2'].spaceAfter = 10
    styles['Heading3'].fontSize = 12
    styles['Heading3'].spaceAfter = 8
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Bold', parent=styles['Normal'], fontName='Helvetica-Bold'))

    content = []

    # 添加标题
    content.append(Paragraph(f"Plant Report: {plant_id}", styles['Heading1']))
    content.append(Paragraph(f"Date: {date}", styles['Heading2']))
    content.append(Spacer(1, 12))

    # 处理报告文本
    for line in report_text.split('\n'):
        line = line.strip()
        if line.startswith('**') and line.endswith('**'):
            # 主标题
            content.append(Paragraph(line.strip('**'), styles['Heading2']))
        elif '**' in line:
            # 处理行内的加粗文本
            parts = line.split('**')
            formatted_line = ''
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    formatted_line += part
                else:
                    formatted_line += f'<b>{part}</b>'
            content.append(Paragraph(formatted_line, styles['Normal']))
        elif line.startswith('-'):
            # 列表项
            content.append(Paragraph(f"• {line[1:].strip()}", styles['Normal']))
        elif line:
            # 普通段落
            content.append(Paragraph(line, styles['Normal']))
        else:
            content.append(Spacer(1, 6))

    content.append(Spacer(1, 24))

    # 添加图片
    for view, image_url in images.items():
        image_path = os.path.join(PLANT_DATA_DIR, image_url.replace("/static/", ""))
        if os.path.exists(image_path):
            content.append(Paragraph(f"Plant Image - {view}", styles['Heading3']))
            # 调整三个侧视图的宽度
            if view.startswith('sv-'):
                img = Image(image_path, width=5 * inch, height=3.75 * inch)
            else:
                img = Image(image_path, width=6 * inch, height=4.5 * inch)
            content.append(img)
            content.append(Spacer(1, 12))

    # 添加分支分析图片
    for view, data in branch_analysis.items():
        if 'frame' in data:
            frame = data['frame']
            folder_path = f"/Users/tshoiasc/Documents/文稿 - 陈跃千纪的MacBook Pro (325)/durham/毕业论文/skeleton/result/{plant_id}-{view}"
            skeleton_image = os.path.join(folder_path, f"frame_{frame:03d}_skeleton_and_summary.png")
            branch_image = os.path.join(folder_path, f"frame_{frame:03d}_individual_paths.png")

            if os.path.exists(skeleton_image) and os.path.exists(branch_image):
                content.append(Paragraph(f"Branch Analysis for {view}", styles['Heading3']))

                img1 = Image(skeleton_image, width=6.5 * inch, height=4 * inch)
                content.append(img1)
                content.append(Spacer(1, 12))

                # 进一步缩短 individual_paths 图片的高度
                img2 = Image(branch_image, width=7 * inch, height=0.8 * inch)
                content.append(img2)
                content.append(Spacer(1, 24))

    # 生成 PDF
    doc.build(content)
    return pdf_path


@app.get("/bud-analysis/{plant_id}/{date}")
async def get_bud_analysis(plant_id: str, date: str):
    try:
        logger.info(f"Fetching bud analysis for plant_id: {plant_id}, date: {date}")

        file_path = Path(f"target_detection/{plant_id}.json")
        logger.info(f"Looking for file: {file_path}")

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"Bud analysis data not found for plant_id: {plant_id}")

        with open(file_path, 'r') as file:
            data = json.load(file)

        logger.info(f"Successfully loaded data for {plant_id}")

        # Filter the data for the specific date
        filtered_data = {}
        for view, view_data in data.items():
            filtered_view_data = [item for item in view_data if date in item['image_name']]
            if filtered_view_data:
                filtered_data[view] = filtered_view_data

        if not filtered_data:
            logger.warning(f"No data found for date: {date}")
            return {"message": f"No data found for date: {date}"}

        # Format the data for frontend use
        formatted_data = {}
        for view, view_data in filtered_data.items():
            formatted_data[view] = []
            for item in view_data:
                if 'predictions' in item and item['predictions']:
                    for prediction in item['predictions']:
                        formatted_data[view].append({
                            'boxes': prediction['boxes'],
                            'conf': prediction['conf'],
                            'cls': prediction['cls']
                        })

        logger.info(f"Returning formatted data for {plant_id}, date: {date}")
        return formatted_data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error decoding JSON file for plant_id: {plant_id}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory=PLANT_DATA_DIR), name="static")


@app.get("/plant-images/{plant_id}/{date}")
async def get_plant_images(plant_id: str, date: str):
    images = {}
    base_dir = os.path.join(PLANT_DATA_DIR, plant_id)
    print(f"Base directory: {base_dir}")  # 添加日志

    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")  # 添加日志
        raise HTTPException(status_code=404, detail=f"Plant directory not found for {plant_id}")

    for view in ["sv-000", "sv-045", "sv-090", "tv-000"]:
        image_dir = os.path.join(base_dir, view)
        filename = f"{plant_id}-{date}_00_VIS_{view.replace('-', '_')}-0-0-0.png"
        full_path = os.path.join(image_dir, filename)
        print(f"Checking for file: {full_path}")  # 添加日志
        if os.path.exists(full_path):
            images[view] = f"/static/{plant_id}/{view}/{filename}"
            print(f"Found image: {images[view]}")  # 添加日志
        else:
            print(f"Image not found: {full_path}")  # 添加日志

    if not images:
        print(f"No images found for plant {plant_id} on date {date}")  # 添加日志
        raise HTTPException(status_code=404, detail=f"Images not found for plant {plant_id} on date {date}")

    print(f"Returning images: {images}")  # 添加日志
    return images


@app.get("/image/{plant_id}/{filename}")
async def get_image(plant_id: str, filename: str):
    image_path = os.path.join(PLANT_DATA_DIR, plant_id, filename)
    print(f"Attempting to serve image: {image_path}")  # 添加日志
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")  # 添加日志
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


@app.get("/plant-info/{plant_id}")
async def get_plant_info(plant_id: str):
    # 这里返回示例数据，实际应用中应该从数据库或文件中读取
    return {
        "variety": "Sample Variety",
        "growthStage": "Flowering"
    }


@app.get("/bud-info/{plant_id}/{date}")
async def get_bud_info(plant_id: str, date: str):
    try:
        logger.info(f"Fetching bud info for plant_id: {plant_id}, date: {date}")

        file_path = Path(f"target_detection/{plant_id}.json")
        logger.info(f"Looking for file: {file_path}")

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"Bud analysis data not found for plant_id: {plant_id}")

        with open(file_path, 'r') as file:
            data = json.load(file)

        logger.info(f"Successfully loaded data for {plant_id}")

        # Filter the data for the specific date
        filtered_data = {}
        for view, view_data in data.items():
            filtered_view_data = [item for item in view_data if date in item['image_name']]
            if filtered_view_data:
                filtered_data[view] = filtered_view_data

        if not filtered_data:
            logger.warning(f"No data found for date: {date}")
            return {
                "hasCurrentDateData": False,
                "budCounts": {}
            }

        # Count buds for each view
        bud_counts = {}
        for view, view_data in filtered_data.items():
            bud_counts[view] = sum(len(item['predictions']) for item in view_data if 'predictions' in item)

        logger.info(f"Returning bud info for {plant_id}, date: {date}")
        return {
            "hasCurrentDateData": True,
            "budCounts": bud_counts
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error decoding JSON file for plant_id: {plant_id}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/branch-info/{plant_id}/{date}")
async def get_branch_info(plant_id: str, date: str):
    logger.info(f"Processing branch info for plant_id: {plant_id}, date: {date}")
    try:
        earliest_date = None
        current_date_data = None
        main_stem_length = 0
        total_branches = 0
        secondary_branches = 0
        tertiary_branches = 0
        has_current_date_data = False

        for view in ['sv-000', 'sv-045', 'sv-090']:
            file_path = f"/Users/tshoiasc/Documents/文稿 - 陈跃千纪的MacBook Pro (325)/durham/毕业论文/skeleton/result/{plant_id}-{view}/branch_and_trajectory_data.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if not df.empty:
                    # 使用正则表达式提取日期
                    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
                    first_date_match = date_pattern.search(df['Image Name'].iloc[0])
                    if first_date_match:
                        first_date_str = first_date_match.group()
                        first_date = datetime.strptime(first_date_str, '%Y-%m-%d')
                        if earliest_date is None or first_date < earliest_date:
                            earliest_date = first_date

                    # 检查当前日期是否有数据
                    current_date_data = df[df['Image Name'].str.contains(date)]
                    if not current_date_data.empty:
                        has_current_date_data = True
                        # 计算分支信息
                        main_stem = current_date_data[current_date_data['Branch Level'] == 1].iloc[0]
                        main_stem_length = main_stem['Length']
                        total_branches = len(current_date_data)
                        secondary_branches = len(current_date_data[current_date_data['Branch Level'] == 2])
                        tertiary_branches = len(current_date_data[current_date_data['Branch Level'] == 3])
                        break  # 如果找到当前日期的数据，就不需要继续查找其他视图

        if earliest_date is None:
            raise HTTPException(status_code=404, detail=f"No data found for plant_id: {plant_id}")

        return {
            "earliestDate": earliest_date.strftime('%Y-%m-%d'),
            "hasCurrentDateData": has_current_date_data,
            "mainStemLength": round(main_stem_length, 2) if has_current_date_data else None,
            "totalBranches": total_branches,
            "secondaryBranches": secondary_branches,
            "tertiaryBranches": tertiary_branches,
        }

    except Exception as e:
        logger.error(f"Unexpected error in get_branch_info: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def handle_nan_inf(obj):
    if isinstance(obj, (np.float, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


@app.get("/plant-details/{plant_id}")
async def get_plant_details(plant_id: str):
    try:
        # 使用正则表达式验证 plant_id 格式
        if not re.match(r'BR017-\d{6}', plant_id):
            raise HTTPException(status_code=400, detail=f"Invalid plant_id format: {plant_id}")

        # 确定正确的sheet名称
        if plant_id[-2] == '1':
            sheet_name = "S1. BR017_5C"
        elif plant_id[-2] == '2':
            sheet_name = "S2. BR017_10C"
        else:
            raise HTTPException(status_code=400, detail=f"Invalid plant_id format: {plant_id}")

        # 读取Excel文件的特定sheet
        try:
            df = pd.read_excel("data/Extant plant data.xlsx", sheet_name=sheet_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading Excel file: {str(e)}")

        # 查找特定plant_id的行
        plant_data = df[df['plant_id'] == plant_id]

        if plant_data.empty:
            # 如果找不到植物，我们打印出所有可用的plant_id以便调试
            available_plants = df['plant_id'].tolist()
            error_msg = f"Plant not found: {plant_id}. Available plant_ids: {available_plants}"
            print(error_msg)  # 打印到控制台
            raise HTTPException(status_code=404, detail=error_msg)

        # 将DataFrame行转换为字典
        plant_dict = plant_data.iloc[0].to_dict()

        # 处理NaN和Inf值
        for key, value in plant_dict.items():
            if isinstance(value, (float, np.float64)):
                if np.isnan(value) or np.isinf(value):
                    plant_dict[key] = None
                else:
                    plant_dict[key] = float(value)
            elif isinstance(value, (int, np.int64)):
                plant_dict[key] = int(value)
            elif isinstance(value, str):
                plant_dict[key] = value
            else:
                plant_dict[key] = str(value)  # 转换其他类型为字符串

        return plant_dict
    except HTTPException as he:
        # 直接重新抛出 HTTPException
        raise he
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # 在服务器控制台打印错误信息
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/branch-analysis/{plant_id}/{date}/{view}")
async def get_branch_analysis(plant_id: str, date: str, view: str):
    try:
        logger.info(f"Fetching branch analysis for plant_id: {plant_id}, date: {date}, view: {view}")

        folder_path = f"/Users/tshoiasc/Documents/文稿 - 陈跃千纪的MacBook Pro (325)/durham/毕业论文/skeleton/result/{plant_id}-{view}"
        file_path = os.path.join(folder_path, "branch_and_trajectory_data.csv")

        logger.info(f"Looking for file: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404,
                                detail=f"Branch data not found for plant_id: {plant_id} and view: {view}")

        df = pd.read_csv(file_path)
        logger.info(f"CSV file loaded. Shape: {df.shape}")

        # 构建要筛选的文件名前缀
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        file_prefix = f"{plant_id}-{date_obj.strftime('%Y-%m-%d')}_00_VIS_{view.replace('-', '_')}"
        logger.info(f"File prefix for filtering: {file_prefix}")

        # 筛选数据
        filtered_df = df[df['Image Name'].str.startswith(file_prefix)]
        logger.info(f"Filtered data shape: {filtered_df.shape}")

        if filtered_df.empty:
            logger.warning(f"No data found for plant_id: {plant_id}, date: {date}, and view: {view}")
            sample_image_names = df['Image Name'].head().tolist()
            logger.info(f"Sample image names: {sample_image_names}")
            return {"message": f"No data found for the specified date. Sample image names: {sample_image_names}"}

        # 获取 Frame 值
        frame = int(filtered_df['Frame'].iloc[0]) if 'Frame' in filtered_df.columns else None
        logger.info(f"Frame value: {frame}")

        branches = []
        for _, row in filtered_df.iterrows():
            try:
                branch = {
                    "level": int(row["Branch Level"]),
                    "bud_position": eval(row["Bud Position"]),
                    "branch_path": [eval(point) for point in row["Branch Path"].split(";")],
                    "angle": float(row["Angle"]) if not pd.isna(row["Angle"]) else None,
                    "length": float(row["Length"]),
                    "vertical_length": float(row["Vertical Length"]),
                    "horizontal_length": float(row["Horizontal Length"])
                }
                branches.append(branch)
            except Exception as e:
                logger.error(f"Error processing row: {row}")
                logger.error(f"Error details: {str(e)}")

        logger.info(f"Returning {len(branches)} branches")
        return {"branches": branches, "frame": frame}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)