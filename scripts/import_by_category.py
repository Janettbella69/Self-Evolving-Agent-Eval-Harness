"""Import ShoppingComp dataset into eval system, split by product category.

Creates one dataset per category, combining EN+ZH and normal+trap cases.
Run from the repo root: python3 scripts/import_by_category.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add eval backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from storage.database import init_db
from storage import queries

# Resolve data dir relative to this script's location (scripts/ under repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "datasets" / "ShoppingComp"

FILES = [
    ("ShoppingComp_97_20260127.en.jsonl", "en", False),
    ("ShoppingComp_97_20260127.zh.jsonl", "zh", False),
    ("ShoppingComp_traps_48_20260127.en.jsonl", "en", True),
    ("ShoppingComp_traps_48_20260127.zh.jsonl", "zh", True),
]

# ── Category classification rules (order matters: first match wins) ──────

CATEGORY_RULES = [
    ("Camera-Photography", [
        "camera", "相机", "photographer", "摄影", "摄像", "lens", "镜头",
        "webcam", "gopro", "tripod", "脚架", "photo/video gig",
    ]),
    ("Phone-Mobile", [
        "phone", "手机", "smartphone", "iphone", "android", "pixel",
    ]),
    ("Laptop-Computer", [
        "laptop", "笔记本", "macbook", "chromebook", "notebook computer",
    ]),
    ("PC-Peripherals", [
        "gpu", "显卡", "graphics card", "pc build", "mini-itx", "cpu",
        "motherboard", "putting together a new pc",
        "keyboard", "键盘", "mouse", "鼠标", "touchpad",
        "game controller", "手柄", "gamepad",
        "printer", "打印",
    ]),
    ("Tablet-E-reader", [
        "tablet", "平板", "e-reader", "kindle", "ipad",
    ]),
    ("TV-Display", [
        " tv ", "television", "电视", "monitor", "显示器",
        "projector", "投影",
    ]),
    ("Audio-Headphones", [
        "headphone", "headset", "earphone", "earbuds", "耳机",
        "speaker", "音箱", "soundbar", "audio", "gaming headset",
        "bluetooth headset", "gaming stream",
    ]),
    ("Home-Appliances", [
        "washing machine", "洗衣机", "refrigerator", "冰箱",
        "air conditioner", "空调", " ac ", "choose an ac",
        "vacuum", "吸尘", "range hood", "油烟机",
        "dehumidifier", "除湿", "water heater", "热水器",
        "hot water", "shower back to back",
        "bathroom heater", "浴霸", "dishwasher", "洗碗机",
        "clothes-drying rack", "晾衣架", "drying rack",
        "air purifier", "净化器", "ceiling fan",
        "toilet", "马桶", "electric blanket", "电热毯",
    ]),
    ("Kitchen", [
        "rice cooker", "电饭", "coffee", "咖啡", "oven", "烤箱",
        "blender", "料理机", "water purifier", "净水",
        "kettle", "热水壶", "juicer", "榨汁", "meat grinder", "绞肉",
        "microwave", "微波", "induction", "电磁炉", "slow cooker",
        "消毒柜", "sterilizer cabinet", "破壁机",
    ]),
    ("Beauty-Skincare", [
        "skincare", "护肤", "sunscreen", "防晒", "foundation", "粉底",
        "cosmetic", "化妆", "serum", "精华", "cleanser", "洁面",
        "moistur", "保湿", "mask", "面膜", "lip ", "唇",
        "exfoliant", "去角质", "blackhead", "黑头", "acne", "痤疮",
        "makeup", "卸妆", "remover", "beauty", "perfume", "fragrance", "香水",
        "hair dryer", "吹风", "hair", "头发", "frizzy",
        "shaver", "剃须", "skin", "rosacea", "laser treatment",
        "hydrating", "补水", "定妆", "setting spray",
        "清洁产品", "皂基", "洗面奶",
        "退热贴",  # baby cooling patch (beauty-adjacent health)
    ]),
    ("Pet-Baby", [
        "pet", "宠物", "baby", "婴儿", "stroller", "推车",
        "car seat", "安全座椅", "dog", "cat", "猫", "狗",
        "diaper", "尿布", "宝宝", "child", "孩子",
    ]),
    ("Furniture-Home", [
        "mattress", "床垫", "sofa", "沙发", "desk", "书桌",
        "chair", "椅", "lighting", "灯", "light in my home studio",
        "bed mite", "除螨",
        "flooring", "地板", "floor", "涂料", "paint",
        "formaldehyde", "甲醛", "活性炭", "renovating", "翻新",
        "防水涂料", "发霉", "mold", "隔墙", "裂纹",
    ]),
    ("Watch-Wearable", [
        "watch", "手表", "wearable", "fitness tracker", "fitbit",
        "garmin", "triathlet",
    ]),
    ("Fashion-Outdoor", [
        "shoe", "鞋", "running", "跑步", "backpack", "背包",
        "luggage", "行李箱", "jacket", "冲锋衣",
        "snowboard", "滑雪", "fishing", "钓鱼",
        "bicycle", "自行车", "bike", "骑行",
        "drone", "无人机", "camping", "帐篷", "hiking",
        "protective gear", "护具",
        "legging", "秋裤", "dopamine dressing",
    ]),
    ("Smart-Home-Networking", [
        "smart lock", "智能锁", "robot", "机器人",
        "smart home", "智能家居", "router", "路由器",
        "wifi", "mesh", "security camera", "监控",
        "sensor", "传感器", "solar", "broadband",
        "quarry", "stone quarry",
    ]),
    ("Automotive", [
        "car ", "汽车", "vehicle", "dash cam", "行车记录仪",
        "engine off", "portable power", "车载",
        "car as a temporary office",
    ]),
    ("Medical-Health", [
        "copd", "bronchitis", "medical", "nebulizer", "blood pressure",
        "health monitor", "oxygen", "制氧",
        "按摩仪", "massage", "cervical spondylosis", "颈椎",
        "肩周炎",
    ]),
]


def classify(question: str) -> str:
    q = question.lower()
    for cat, keywords in CATEGORY_RULES:
        for kw in keywords:
            if kw in q:
                return cat
    return "Other"


async def main():
    await init_db()

    # Read all entries and classify
    categorized: dict[str, list[dict]] = {}  # cat -> [entries with metadata]

    for fname, lang, is_trap in FILES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"SKIP: {path} not found")
            continue

        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                cat = classify(entry["question"])
                if cat not in categorized:
                    categorized[cat] = []
                categorized[cat].append({
                    "entry": entry,
                    "lang": lang,
                    "is_trap": is_trap,
                })

    # Print summary
    print(f"\n{'Category':<25s} {'Count':>6s}")
    print("-" * 35)
    total = 0
    for cat in sorted(categorized, key=lambda c: -len(categorized[c])):
        n = len(categorized[cat])
        total += n
        print(f"  {cat:<23s} {n:>5d}")
    print("-" * 35)
    print(f"  {'TOTAL':<23s} {total:>5d}")

    # Create datasets
    print(f"\nCreating {len(categorized)} category datasets...\n")

    for cat in sorted(categorized, key=lambda c: -len(categorized[c])):
        items = categorized[cat]
        dataset_name = f"SC-{cat}"

        # Count by type
        en_normal = sum(1 for i in items if i["lang"] == "en" and not i["is_trap"])
        zh_normal = sum(1 for i in items if i["lang"] == "zh" and not i["is_trap"])
        en_trap = sum(1 for i in items if i["lang"] == "en" and i["is_trap"])
        zh_trap = sum(1 for i in items if i["lang"] == "zh" and i["is_trap"])

        description = (
            f"ShoppingComp {cat.replace('-', '/')} "
            f"({en_normal}EN + {zh_normal}ZH + {en_trap}EN-trap + {zh_trap}ZH-trap)"
        )

        # Check if dataset exists
        existing = await queries.get_dataset_by_name(dataset_name)
        if existing:
            dataset_id = existing.id
            print(f"  EXISTS: {dataset_name} (id={dataset_id}), updating cases...")
        else:
            dataset_id = await queries.create_dataset(
                dataset_name, description, "capability"
            )
            print(f"  CREATE: {dataset_name} (id={dataset_id})")

        imported = 0
        for item in items:
            entry = item["entry"]
            lang = item["lang"]
            is_trap = item["is_trap"]

            uuid = entry.get("uuid", "")
            question = entry.get("question", "")
            if not uuid or not question:
                continue

            # Case key: lang prefix + uuid[:8]
            trap_tag = "T" if is_trap else ""
            case_key = f"{lang.upper()}{trap_tag}-{uuid[:8]}"

            # Determine case type
            has_scenes = bool(entry.get("scene_list"))
            has_trap_rubric = bool(entry.get("trap_rubric"))

            if has_trap_rubric:
                case_type = "trap"
                golden_data = {
                    "trap_rubric": entry["trap_rubric"],
                }
            elif has_scenes:
                case_type = "shoppingcomp"
                golden_data = {
                    "scene_list": entry.get("scene_list", []),
                    "product_list": entry.get("product_list", []),
                }
            else:
                case_type = "clear_en" if lang == "en" else "clear_zh"
                golden_data = {}

            if entry.get("source"):
                golden_data["source"] = entry["source"]

            await queries.upsert_case(
                dataset_id=dataset_id,
                key=case_key,
                query=question,
                case_type=case_type,
                constraints={},
                golden_data=golden_data,
            )
            imported += 1

        print(f"          → {imported} cases imported")

    print(f"\nDone! {len(categorized)} datasets created.")


if __name__ == "__main__":
    asyncio.run(main())
