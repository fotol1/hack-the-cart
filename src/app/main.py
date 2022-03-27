from typing import List
import json
import numpy as np
from pathlib import Path
from http import HTTPStatus
from functools import lru_cache
from pydantic import BaseModel, Field
from fastapi import FastAPI, Response, Depends


class Model:
    def __init__(self, item_2_idx_path: Path, weights_path: Path) -> None:
        with item_2_idx_path.open("r", encoding="utf-8") as file:
            self._item_2_idx = json.load(file)
            self._idx_2_item = {v: k for k, v in self._item_2_idx.items()}
        self._weights = np.load(weights_path)
        self._num_items = len(self._item_2_idx)

    def predict(self, items: List[str], topk: int = 10) -> List[str]:
        # Build items array
        items = [item for x in items if (item := self._item_2_idx.get(x)) is not None]
        items_array = np.zeros(self._num_items)
        items_array[items] = 1.0
        # Get scores
        scores = items_array.dot(self._weights)
        scores[items_array > 0] = -np.inf
        predicted_items = np.argsort(-scores)[-topk:]
        # Decode items
        return [item for x in predicted_items if (item := self._idx_2_item.get(x)) is not None]


@lru_cache(maxsize=None)
def get_model() -> Model:
    data_dir = Path.cwd() / "src" / "app" / "data"
    return Model(
        item_2_idx_path=data_dir / "item_to_idx.json",
        weights_path=data_dir / "weights.npy",
    )


async def model_dependency() -> Model:
    return get_model()


async def health() -> Response:
    return Response(content="Ok", status_code=HTTPStatus.OK)


class RecsResponse(BaseModel):
    status: str = Field(default="success", description="Request status: success, error.")
    message: str = Field(
        default=None,
        description="Response message. It is present if an error occurres.",
    )
    output: List[str] = Field(default=[], description="List of recommendations for user.")


async def get_recs(query: str, model: Model = Depends(model_dependency)) -> RecsResponse:
    items = query.split("__")
    return RecsResponse(output=model.predict(items))


def get_app() -> FastAPI:
    # Get model in cache
    get_model()
    app = FastAPI(description=(Path.cwd() / "README.md").read_text())
    app.add_api_route(
        "/recs",
        get_recs,
        methods=["GET"],
        description="Get recommendations for user' items.",
        response_model=RecsResponse,
    )
    app.add_api_route("/health", health, methods=["GET"], description="I'm alive!!!!!")
    return app
