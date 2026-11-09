"""只读核验既有官方 Instruct 模型；不调用 API、不注册或下载模型。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.ollama_model_identity import verify_model_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_model_store(args.models_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
